from gemma.utils import bilby_conversion as bilby_conv
from gemma.backend import get_arraylib, jit_if_available
from gemma.ejecta.functions import calculate_ns_compactness, calculate_baryonic_mass, calculate_r_isco

xp = get_arraylib()
jit = jit_if_available()

# More reliable check
is_jax_backend = xp.__name__ == "jax.numpy"

def flexible_jit(static_argnames=()):
    def decorator(fn):
        if is_jax_backend:
            return jit(fn, static_argnames=static_argnames)
        else:
            return jit(fn)
    return decorator

@flexible_jit(static_argnames=["dynamical_mass", "dynamical_velocity", "disk_mass", "preprocessed"])
def nsbh_properties(mass_1=None,
                    mass_2=None,
                    lambda_2=None,
                    chi_1=None,
                    chirp_mass=None,
                    mass_ratio=None,
                    quarks=False,
                    fix_xsi=True,
                    xsi_override=0.0,
                    use_mosfit_model=False,# Dummy for now
                    use_uncertainty=False,
                    key=None,
                    dynamical_mass=None,
                    dynamical_velocity=None,
                    disk_mass=None,
                    preprocessed=False):
    """
    Functional and JIT-safe version of NSBH ejecta property computation.

    Parameters:
        mass_1, mass_2: Component masses [Msun]
        lambda_2: NS tidal deformability
        chi_1: Dimensionless spin of the BH
        fix_xsi: Whether to compute xsi from Eq. 12 or use override
        xsi_override: Fixed value to use if not fixing
        use_uncertainty: Whether to sample errors
        formula_index_dyn: Index for mass model: 0 = kruger, 1 = kawaguchi
        key: PRNGKey (JAX) if use_uncertainty is True

    Returns:
        dict with keys: m_dyn, m_disk, m_wind, v_dyn, m_jet
    """
    if not preprocessed:
        # Handle cases where mass_1, mass_2, mass_ratio, or chirp_mass are provided
        if mass_1 is not None and mass_2 is not None:
            mass_ratio = mass_2 / mass_1
            chirp_mass = bilby_conv.component_masses_to_chirp_mass(mass_1, mass_2)
        elif mass_ratio is not None and chirp_mass is not None:
            mass_1, mass_2 = bilby_conv.chirp_mass_and_mass_ratio_to_component_masses(chirp_mass, mass_ratio)


    # Calculate other parameters
    Q = mass_1 / mass_2
    mass_ratio = mass_2 / mass_1
    eta = mass_1 * mass_2 / (mass_1 + mass_2)**2

    c2 = calculate_ns_compactness(lambda_2, quarks=quarks)
    mb2 = calculate_baryonic_mass(mass_2, c2)
    r_isco = calculate_r_isco(chi_1)

    # Fitting functions

    # Dynamical ejecta mass (model-dependent)
    if dynamical_mass == "kruger":
        m_dyn = calculate_bhns_dynamical_mass_kruger(Q, c2, r_isco)

    elif dynamical_mass == "kawaguchi":
        m_dyn = calculate_bhns_dynamical_mass_kawaguchi(Q, c2, r_isco, mass_2, mb2)
    else:
        raise ValueError(f"Unsupported dynamical_mass: calculate_bhns_dynamical_mass_{dynamical_mass}")

    # Dynamical velocity (model-dependent)
    if dynamical_velocity == "foucart":
        v_dyn = calculate_bhns_dynamical_velocity_foucart(Q)

    elif dynamical_velocity == "kawaguchi":
        v_dyn = calculate_bhns_dynamical_velocity_kawaguchi(Q)

    else:
        raise ValueError(f"Unsupported dynamical_velocity: calculate_bhns_dynamical_velocity_{dynamical_velocity}")

    # Disk mass (model-dependent)
    if disk_mass == "foucart":
        m_disk = calculate_bhns_disk_mass_foucart(c2, eta, r_isco, mb2, m_dyn)

    else:
        raise ValueError(f"Unsupported disk_mass: calculate_bhns_disk_mass_{disk_mass}")

    xsi = (
        0.18 + (0.29 - 0.18) / (1.0 + xp.exp(1.5 * (mass_1 / mass_2 - 3.0)))
        if fix_xsi else xsi_override)

    if use_uncertainty:
        assert key is not None, "PRNG key must be provided if using uncertainty."
        err_props = nsbh_errors_properties(
            m_dyn, m_disk, v_dyn, Q, mb2,
            formula_index_dyn,
            not fix_xsi, xsi_override,
            key
        )
        m_dyn = err_props["m_dyn"]
        m_disk = err_props["m_disk"]
        v_dyn = err_props["v_dyn"]
        m_wind = err_props["m_wind"]
    else:
        m_wind = xsi * m_disk
    return {"m_ejecta_dyn": m_dyn,
            "m_ejecta_disk": m_disk,
            "m_ejecta_wind": m_wind,
            "v_ejecta_dyn": v_dyn,
            "m_ejecta_jet": m_disk - m_dyn,
            "m_tov": 0.0,
            "threshold_mass":0.0,
            "compactness_1": 0.0,
            "compactness_2": c2}

    # return {"m_dyn": m_dyn, "m_disk": m_disk, "m_wind": m_wind, "v_dyn": v_dyn, "m_jet": m_disk - m_dyn,
    #         "m_tov":None, "threshold_mass":None, "compactness_1":0., "compactness_2": c2 }
    #


@jit
def nsbh_errors_properties(
    m_dyn, m_disk, v_dyn, Q,
    m_baryonic_2, formula_index,  # 0 = kruger, 1 = kawaguchi
    xsi_provided: bool, xsi_value: float,
    key):
    """
    JIT-compatible uncertainty sampling for NSBH ejecta properties (JAX version).

    Parameters:
        m_dyn, m_disk, v_dyn: Ejecta properties [Msun, Msun, dimensionless]
        Q: Mass ratio m1/m2
        m_baryonic_2: Baryonic mass of neutron star [Msun]
        formula_index: 0 = kruger, 1 = kawaguchi
        xsi_provided: If True, use xsi_value
        xsi_value: Fixed xsi value if provided
        key: jax.random.PRNGKey

    Returns:
        Dictionary with sampled: m_dyn, m_disk, v_dyn, m_wind
    """
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # Dynamical mass error: 10% relative + constant
    dyn_error_scale = jnp.where(formula_index == 0, 0.01, 0.005)
    err_m_dyn = jnp.sqrt((0.1 * m_dyn)**2 + dyn_error_scale**2)

    err_m_disk = m_baryonic_2 * jnp.sqrt((0.1 * m_disk)**2 + 0.01**2)
    err_v_dyn = 0.1 * v_dyn

    sampled_m_dyn = jax.random.normal(k1, ()) * err_m_dyn + m_dyn
    sampled_m_disk = jax.random.normal(k2, ()) * err_m_disk + m_disk
    sampled_v_dyn = jax.random.normal(k3, ()) * err_v_dyn + v_dyn

    m_dyn_out = jnp.maximum(0.0, sampled_m_dyn)
    m_disk_out = jnp.maximum(0.0, sampled_m_disk)
    v_dyn_out = jnp.maximum(0.0, sampled_v_dyn)

    # xsi sampling if not provided
    xsi_low = 0.04 + (0.14 - 0.04) / (1.0 + jnp.exp(1.5 * (Q - 3.0)))
    xsi_up = 0.32 + (0.44 - 0.32) / (1.0 + jnp.exp(1.5 * (Q - 3.0)))
    sampled_xsi = jax.random.uniform(k4, (), minval=xsi_low, maxval=xsi_up)

    xsi_final = jnp.where(xsi_provided, xsi_value, sampled_xsi)
    m_wind = xsi_final * m_disk_out

    return {"m_dyn": m_dyn_out, "m_disk": m_disk_out, "v_dyn": v_dyn_out, "m_wind": m_wind}



@jit
def calculate_bhns_disk_mass_foucart(c2, eta, r_isco, mb2, m_dyn):
    """
    Calculate the disk mass for a BHNS system using the Foucart et al. (2018) model.

    Parameters:
        c2: Compactness of the neutron star
        eta: Symmetric mass ratio
        r_isco: Innermost stable circular orbit radius [GM/c^2]
        mb2: Baryonic mass of the neutron star [Msun]
        m_dyn: Dynamical ejecta mass [Msun]

    Returns:
        Disk mass [Msun]
    """
    # Model coefficients from Foucart et al. (2018)
    alpha = 0.406
    beta = 0.139
    gamma = 0.255
    delta = 1.761

    # Fit formula for the remnant mass (before subtracting ejecta)
    remnant_mass = (
        alpha * (1.0 - 2.0 * c2) / eta**(1.0 / 3.0) -
        beta * r_isco * c2 / eta + gamma)

    remnant_mass = xp.maximum(remnant_mass, 0.0) ** delta * mb2

    # Final disk mass after subtracting dynamical ejecta
    return xp.maximum(remnant_mass - m_dyn, 0.0)

@jit
def calculate_bhns_dynamical_mass_kruger(Q, c2, r_isco):
    """
    Calculate the dynamical ejecta mass for a BHNS system using the
    Kruger & Foucart (2020) model.

    Parameters:
        Q: Inverse mass ratio m1/m2 (BH mass / NS mass), dimensionless
        c2: Compactness of the neutron star
        r_isco: ISCO radius [GM/c^2]

    Returns:
        Dynamical ejecta mass [Msun]
    """
    # Fit coefficients from Kruger & Foucart (2020)
    a1 = 7.11595154e-03
    a2 = 1.43636803e-03
    a4 = -2.76202990e-02
    n1 = 0.863604211
    n2 = 1.68399507

    # Fit expression
    m_dyn = (a1 * Q**n1 * (1.0 - 2.0 * c2) / c2 - a2 * Q**n2 * r_isco + a4)

    return xp.maximum(m_dyn, 0.0)

@jit
def calculate_bhns_dynamical_mass_kawaguchi(Q, compactness_2, r_isco, m2, m_baryonic_2):
    """
    Calculate the dynamical ejecta mass for a BHNS system using the Kawaguchi et al. (2016) model.

    Parameters:
        Q: Inverse mass ratio m1/m2 (BH mass / NS mass), dimensionless
        compactness_2: Compactness of the neutron star
        r_isco: ISCO radius [GM/c^2]
        m2: Gravitational mass of the neutron star [Msun]
        m_baryonic_2: Baryonic mass of the neutron star [Msun]

    Returns:
        Dynamical ejecta mass [Msun]
    """
    # Coefficients from Kawaguchi et al. 2016
    a1 = 0.04464
    a2 = 0.002269
    a3 = 2.431
    a4 = -0.4159
    n1 = 0.2497
    n2 = 1.352

    # Fitting formula
    m_dyn = (
        a1 * Q**n1 * (1.0 - 2.0 * compactness_2) / compactness_2
        - a2 * Q**n2 * r_isco
        + a3 * (1.0 - m2 / m_baryonic_2)
        + a4
    )

    return xp.maximum(m_dyn, 0.0)


@jit
def calculate_bhns_dynamical_velocity_foucart(Q):
    """
    Calculate the dynamical ejecta velocity for a BHNS system using the
    Foucart et al. (2017) model.

    Parameters:
        Q: Mass ratio m1/m2 (BH mass / NS mass), dimensionless

    Returns:
        Dynamical ejecta velocity (dimensionless, in units of c)
    """
    return 0.0149 * Q + 0.1493

@jit
def calculate_bhns_dynamical_velocity_kawaguchi(Q):
    """
    Calculate the dynamical ejecta velocity for a BHNS system using the
    Foucart et al. (2017) model.

    Parameters:
        Q: Mass ratio m1/m2 (BH mass / NS mass), dimensionless

    Returns:
        Dynamical ejecta velocity (dimensionless, in units of c)
    """
    return 0.01533*Q + 0.1907
