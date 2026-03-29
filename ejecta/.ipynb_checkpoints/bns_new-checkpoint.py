from gemma.utils import bilby_conversion as bilby_conv
from gemma.backend import get_arraylib, jit_if_available
from gemma.ejecta.functions import calculate_ns_compactness, calculate_baryonic_mass

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


@flexible_jit(static_argnames=["dynamical_mass", "dynamical_velocity", "disk_mass", "error_func","preprocessed"])
def bns_properties(mass_1=None,
                    mass_2=None,
                    lambda_1=None,
                    lambda_2=None,
                    chirp_mass=None,
                    mass_ratio=None,
                    lambda_tilde=None,
                    fix_xsi=True,
                    xsi_override=0.0,
                    m_tov=2.1,
                    quarks=False,
                    systematics_fraction=0.0,
                    use_mosfit_model=False,
                    use_uncertainty=False,
                    err_v_dyn=0.01,
                    err_m_disk=0.004,
                    formula_index=0,  # e.g. Radice=0, Kruger=1
                    key=None,
                    dynamical_mass=None,
                    dynamical_velocity=None,
                    disk_mass=None,
                    error_func=None,
                    preprocessed=False):
    """
    Functional, JIT-compatible version of BNS ejecta computation.

    Parameters:
        mass_1, mass_2: NS masses [Msun]
        lambda_1, lambda_2: Tidal deformabilities
        chirp_mass: Chirp mass [Msun]
        lambda_tilde: Effective tidal deformability
        fix_xsi: Whether to compute xsi from Raaijmakers+2021 or use override
        xsi_override: User-defined xsi if not fixed
        m_tov: Maximum TOV mass [Msun]
        use_mosfit_model: Whether to apply De+2018 approximation
        use_uncertainty: Whether to include sampling
        err_v_dyn, err_m_disk: Used if uncertainty is enabled
        formula_index: Index for error model
        key: JAX PRNGKey (if uncertainty is used)
        *_func: Callable models for ejecta fitting

    Returns:
        dict with keys: mdyn, mdisk, mwind, vdyn
    """
    if not preprocessed:
        # Handle cases where mass_1, mass_2, mass_ratio, or chirp_mass are provided
        if mass_1 is not None and mass_2 is not None:
            mass_ratio = mass_2 / mass_1
            chirp_mass = bilby_conv.component_masses_to_chirp_mass(mass_1, mass_2)
        elif mass_ratio is not None and chirp_mass is not None:
            mass_1, mass_2 = bilby_conv.chirp_mass_and_mass_ratio_to_component_masses(chirp_mass, mass_ratio)

        # Check how tidal deformability is provided
        if lambda_tilde is not None:
            lambda_1, lambda_2 = bilby_conv.lambda_tilde_to_lambda_1_lambda_2(
                lambda_tilde, mass_1, mass_2)
        else:
            lambda_tilde = bilby_conv.lambda_1_lambda_2_to_lambda_tilde(
                lambda_1, lambda_2, mass_1, mass_2)

    # Calculate other parameters
    Q = mass_1 / mass_2
    eta = mass_1 * mass_2 / (mass_1 + mass_2)**2

    c1 = calculate_ns_compactness(lambda_1, quarks, systematics_fraction)
    c2 = calculate_ns_compactness(lambda_2, quarks, systematics_fraction)
    mb1 = calculate_baryonic_mass(mass_1, c1)
    mb2 = calculate_baryonic_mass(mass_2, c2)

    # Compute the approximate radius of the neutron star
    approx_radius_ns = xp.where(use_mosfit_model, 11.2 * chirp_mass * (lambda_tilde / 800)**(1 / 6),
        chirp_mass * (lambda_tilde / 0.0042)**(1 / 6))

    total_mass = mass_1 + mass_2
    threshold_mass = (2.38 - 3.606 * (m_tov / approx_radius_ns)) * m_tov

    # Compute xsi (either fixed or using the formula)
    xsi = xp.where(fix_xsi,
        0.18 + (0.29 - 0.18) / (1.0 + xp.exp(1.5 * (mass_1 / mass_2 - 3.0))),
        xsi_override)

    # Dynamical ejecta mass (model-dependent)
    if dynamical_mass == "kruger":
        m_dyn = calculate_bns_dynamical_mass_kruger(mass_1, mass_2, c1, c2, mass_ratio, Q)

    elif dynamical_mass == "radice":
        m_dyn = calculate_bns_dynamical_mass_radice(mass_1, mass_2, c1, c2, mb1, mb2)

    elif dynamical_mass == "coughlin":
        m_dyn = calculate_bns_dynamical_mass_coughlin(mass_1, mass_2, c1, c2)

    elif dynamical_mass == "dietrich":
        m_dyn = calculate_bns_dynamical_mass_dietrich(mass_1, mass_2, c1, c2, mb1, mb2)

    elif dynamical_mass == "nedora":
        m_dyn = calculate_bns_dynamical_mass_nedora(mass_1, mass_2, lambda_tilde)

    else:
        raise ValueError(f"Unsupported dynamical_mass: calculate_bns_dynamical_mass_{dynamical_mass}")

    # Dynamical velocity (model-dependent)
    if dynamical_velocity == "radice":
        v_dyn = calculate_bns_dynamical_velocity_radice(c1, c2, mass_ratio, Q)

    elif dynamical_velocity == "coughlin":
        v_dyn = calculate_bns_dynamical_velocity_coughlin(c1, c2, mass_ratio, Q)

    elif dynamical_velocity == "kruger":
        v_dyn = calculate_bns_dynamical_velocity_kruger(mass_ratio)

    else:
        raise ValueError(f"Unsupported dynamical_velocity: calculate_bns_dynamical_velocity_{dynamical_velocity}")

    # Disk mass (model-dependent)
    if disk_mass == "kruger":
        m_disk = calculate_bns_disk_mass_kruger(mass_2, c2)

    elif disk_mass == "radice":
        m_disk = calculate_bns_disk_mass_radice(lambda_tilde)

    elif disk_mass == "coughlin":
        m_disk = calculate_bns_disk_mass_coughlin(total_mass, threshold_mass)

    elif disk_mass == "dietrich":
        m_disk = calculate_bns_disk_mass_dietrich(mass_ratio, total_mass, threshold_mass)

    elif disk_mass == "barbieri":
        m_disk = calculate_bns_disk_mass_barbieri(mass_1, mass_2, lambda_tilde)

    elif disk_mass == "nedora":
        m_disk = calculate_bns_disk_mass_nedora(mass_1, mass_2, lambda_tilde)

    else:
        raise ValueError(f"Unsupported disk_mass: calculate_bns_disk_mass_{disk_mass}")




    if use_uncertainty:
        # If uncertainty sampling is enabled, use the provided error function
        assert key is not None and error_func is not None
        sampled = error_func(
            m_dyn, v_dyn, m_disk, Q,
            err_v_dyn, err_m_disk,
            formula_index,
            not fix_xsi, xsi_override,
            key
        )
        m_dyn = sampled["m_dyn"]
        v_dyn = sampled["v_dyn"]
        m_disk = sampled["mdisk_low"]
        m_wind = sampled["m_wind"]
    else:
        m_wind = xsi * m_disk

    #print("Yves", {"m_dyn": m_dyn,"m_disk": m_disk, "m_wind": m_wind,"v_dyn": v_dyn})

    return {"m_ejecta_dyn": m_dyn,"m_ejecta_disk": m_disk, "m_ejecta_wind": m_wind,"v_ejecta_dyn": v_dyn,
            "m_tov":m_tov, "threshold_mass":threshold_mass, "compactness_1":c1, "compactness_2": c2}



@jit
def bns_errors_properties( m_dyn, v_dyn, m_disk, Q, err_v_dyn, err_m_disk,
    formula_index,  # 0 = radice, 1 = kruger, 2 = coughlin
    xsi_provided: bool, xsi_value: float,
    key):
    """
    JAX-compatible, jit-safe uncertainty sampler for BNS ejecta properties.

    Parameters:
        m_dyn, v_dyn, m_disk: Baseline ejecta properties
        Q: Mass ratio (m1/m2)
        err_v_dyn, err_m_disk: Standard deviations for sampling
        formula_index: Index to select error model (0 = radice, 1 = kruger, 2 = coughlin)
        xsi_provided: bool, whether user specified 'xsi'
        xsi_value: float, fixed xsi if provided
        key: jax.random.PRNGKey

    Returns:
        Dictionary with keys: m_dyn, v_dyn, mdisk_low, m_wind
    """
    k1, k2, k3, k4 = jax.random.split(key, 4)

    err_m_dyn = xp.where(formula_index == 0, 0.5 * m_dyn + 5e-5, 0.004)

    sampled_m_dyn = jax.random.uniform(k1, (), minval=m_dyn - err_m_dyn, maxval=m_dyn + err_m_dyn)
    sampled_v_dyn = jax.random.uniform(k2, (), minval=v_dyn - err_v_dyn, maxval=v_dyn + err_v_dyn)
    sampled_m_disk = jax.random.uniform(k3, (), minval=m_disk - err_m_disk, maxval=m_disk + err_m_disk)

    xsi_low = 0.04 + (0.14 - 0.04) / (1.0 + xp.exp(1.5 * (Q - 3.0)))
    xsi_up = 0.32 + (0.44 - 0.32) / (1.0 + xp.exp(1.5 * (Q - 3.0)))
    sampled_xsi = jax.random.uniform(k4, (), minval=xsi_low, maxval=xsi_up)

    xsi = xp.where(xsi_provided, xsi_value, sampled_xsi)

    return { "m_dyn": xp.maximum(0.0, sampled_m_dyn), "v_dyn": xp.maximum(0.0, sampled_v_dyn), "mdisk_low": xp.maximum(0.0, sampled_m_disk),
        "m_wind": xsi * m_disk}



@jit
def calculate_bns_dynamical_mass_kruger(m1, m2, c1, c2, mass_ratio, q):
    """
    Calculate the dynamical ejecta mass for a BNS system using the Kruger et al. (2020) model.

    Parameters:
        m1: Gravitational mass of the first neutron star [Msun]
        m2: Gravitational mass of the second neutron star [Msun]
        c1: Compactness of the first neutron star
        c2: Compactness of the second neutron star
        mass_ratio: m2 / m1
        q: m1 / m2 (inverse of mass_ratio)

    Returns:
        Dynamical ejecta mass [Msun]
    """
    # Fit parameters
    a, b, c, n = -9.33347796, 114.16916699, -337.55577457, 1.54648799

    term1 = (a / c1 + b * mass_ratio**n + c * c1) * m1
    term2 = (a / c2 + b * q**n + c * c2) * m2

    m_dyn = term1 + term2
    return xp.maximum(m_dyn * 0.001, 1e-6)

@jit
def calculate_bns_dynamical_mass_coughlin(m1, m2, c1, c2):
    """
    Calculate the dynamical ejecta mass for a BNS system using the Coughlin et al. (2019) model.

    Parameters:
        m1: Gravitational mass of the first neutron star [Msun]
        m2: Gravitational mass of the second neutron star [Msun]
        c1: Compactness of the first neutron star
        c2: Compactness of the second neutron star

    Returns:
        Dynamical ejecta mass [Msun]
    """
    # Fitting parameters from Coughlin et al. (2019)
    a = -0.0719
    b = 0.2116
    d = -2.42
    n = -2.905

    # Logarithmic fit formula for dynamical mass
    log_m_dyn = (
        a * (1.0 - 2.0 * c1) / c1 * m1 +
        b * m2 * (m1 / m2) ** n +
        a * (1.0 - 2.0 * c2) / c2 * m2 +
        b * m1 * (m2 / m1) ** n + d
    )

    return 10.0 ** log_m_dyn

@jit
def calculate_bns_dynamical_mass_radice(mass_1, mass_2, compactness_1, compactness_2, m_baryonic_1, m_baryonic_2):
    """
    Calculate the dynamical ejecta mass for a BNS system using the Radice model.

    Parameters:
        mass_1: Gravitational mass of primary [Msun]
        mass_2: Gravitational mass of secondary [Msun]
        m_baryonic_1: Baryonic mass of primary [Msun]
        m_baryonic_2: Baryonic mass of secondary [Msun]
        compactness_1: Compactness of primary
        compactness_2: Compactness of secondary

    Returns:
        Dynamical ejecta mass [Msun]
    """
    # fitting parameters
    a, b, c, d, n = -0.657, 4.254, -32.61, 5.205, -0.773

    q = mass_2 / mass_1
    qinv = mass_1 / mass_2

    term1 = (a * q**(1.0 / 3.0) * (1.0 - 2.0 * compactness_1) / compactness_1
        + b * q**n
        + c * (1.0 - mass_1 / m_baryonic_1)) * m_baryonic_1

    term2 = (a * qinv**(1.0 / 3.0) * (1.0 - 2.0 * compactness_2) / compactness_2
        + b * qinv**n
        + c * (1.0 - mass_2 / m_baryonic_2)) * m_baryonic_2

    m_dyn = term1 + term2 + d

    # Convert to Msun and clip to be >= 0 using where (avoiding xp.clip)
    return xp.maximum(0.001*m_dyn, 0.0)


@jit
def calculate_bns_dynamical_mass_dietrich(m1, m2, c1, c2, mb1, mb2):
    """
    Calculate the dynamical mass for a BNS system using the Dietrich et al. (2016) model.

    Parameters:
        m1: Gravitational mass of the first neutron star [Msun]
        m2: Gravitational mass of the second neutron star [Msun]
        c1: Compactness of the first neutron star
        c2: Compactness of the second neutron star
        mb1: Baryonic mass of the first neutron star [Msun]
        mb2: Baryonic mass of the second neutron star [Msun]

    Returns:
        Dynamical ejecta mass [Msun]
    """
    # Fit coefficients
    a = -1.35695
    b = 6.11252
    c = -49.43355
    d = 16.1144
    n = -2.5484

    # Symmetric dynamical ejecta formula
    term1 = (a * (m2 / m1) ** (1. / 3.) * (1. - 2. * c1) / c1 + b * (m2 / m1) ** n) * mb1
    term2 = (a * (m1 / m2) ** (1. / 3.) * (1. - 2. * c2) / c2 + b * (m1 / m2) ** n) * mb2
    term3 = c * 1.2 * c1 * m1 / (2. - c1)
    term4 = c * 1.2 * c2 * m2 / (2. - c2)

    return  0.001*xp.maximum(term1 + term2 + term3 + term4 + d, 1.0) # Convert to Msun



@jit
def calculate_bns_dynamical_mass_nedora(m1, m2, lambda_tilde):
    """
    Calculate the dynamical ejecta mass for a BNS system using Nedora et al. (2021).

    Parameters:
        m1: gravitational mass of first NS [Msun]
        m2: gravitational mass of second NS [Msun]
        lambda_tilde: effective tidal deformability (dimensionless)

    Returns:
        Dynamical ejecta mass [Msun]
    """
    # Coefficients from Nedora et al. 2021 fit
    b0 = -1.32
    b1 = -3.82e-1
    b2 = -4.47e-3
    b3 = -3.39e-1
    b4 = 3.21e-3
    b5 = 4.31e-7

    # Ensure q ≥ 1 using xp.where for JAX/Numba compatibility
    q = xp.where(m1 > m2, m1 / m2, m2 / m1)

    # Compute log10 of dynamical mass
    log10_mdynamical = (b0 + b1 * q + b2 * lambda_tilde +
                        b3 * q**2 + b4 * lambda_tilde * q + b5 * lambda_tilde**2)

    return 10.0 ** log10_mdynamical

@jit
def calculate_bns_disk_mass_kruger(m2, c2):
    """
    Calculate the disk mass for a BNS system using the Kruger et al. (2020) model.

    Parameters:
        m2: Gravitational mass of the lighter neutron star [Msun]
        c2: Compactness of the lighter neutron star

    Returns:
        Disk mass [Msun]
    """
    # Fitting parameters from Kruger et al. (2020)
    a, c, d = -8.1324, 1.4820, 1.7784

    # Safe minimum for the fit output (non-zero fallback)
    fit_base = a * c2 + c
    base = xp.maximum(fit_base, 5e-4)

    return m2 * base**d



@jit
def calculate_bns_disk_mass_coughlin(total_mass, threshold_mass):
    """
    Calculate the disk mass for a BNS system using the Coughlin et al. (2019) model.

    Parameters:
        total_mass: Total gravitational mass of the BNS system [Msun]
        threshold_mass: Threshold mass for prompt collapse [Msun]

    Returns:
        Disk mass [Msun]
    """
    # Coefficients from the Coughlin 2019 fit
    a = -31.335
    b = -0.976
    c = 1.0474
    d = 0.05957

    # Evaluate log10 remnant disk mass
    remnant_mass_log10 = a * (1 + b * xp.tanh((c - total_mass / threshold_mass) / d))

    # Clip and exponentiate to get physical mass
    remnant_mass_log10 = xp.maximum(remnant_mass_log10, -3.0)
    return 10.0 ** remnant_mass_log10


@jit
def calculate_bns_disk_mass_radice(lambda_tilde):
    """
    Calculate the disk mass for a BNS system using the Radice et al. (2018) model.

    Parameters:
        lambda_tilde: Tidal deformability parameter (dimensionless)

    Returns:
        Disk mass [Msun]
    """
    # Fit parameters from Radice et al. 2018
    alpha = 0.084
    beta = 0.127
    gamma = 567.1
    delta = 405.14

    m_disk = alpha + beta * xp.tanh((lambda_tilde - gamma) / delta)
    return xp.maximum(m_disk, 0.001)
@jit
def calculate_bns_disk_mass_dietrich(mass_ratio, total_mass, threshold_mass):
    """
    Calculate the disk mass for a BNS system using the Dietrich et al. (2020) model.

    Parameters:
        mass_ratio: Mass ratio (m2 / m1), dimensionless
        total_mass: Total gravitational mass of the binary [Msun]
        threshold_mass: Threshold mass for prompt collapse [Msun]

    Returns:
        Disk mass [Msun]
    """
    # Disk wind suppression hyperbolic fit
    beta = 3.910
    qtrans = 0.9
    xsi = 0.5 * xp.tanh(beta * (mass_ratio - qtrans))

    # Coefficients dependent on xsi
    a0 = -1.581
    da = -2.439
    b0 = -0.538
    db = -0.406
    c = 0.953
    d = 0.0417

    # Compute the remnant mass in log10 units
    remnant_mass_log10 = (a0 + da * xsi) * (1 + (b0 + db * xsi) * xp.tanh((c - total_mass / threshold_mass) / d))

    # Clip to prevent extreme negatives and exponentiate
    remnant_mass_log10 = xp.maximum(remnant_mass_log10, -3.0)
    return 10.0 ** remnant_mass_log10



@jit
def calculate_bns_disk_mass_barbieri(m1, m2, lambda_tilde):
    """
    Calculate the disk mass for a BNS system using the Barbieri et al. (2021) model.

    Parameters:
        m1: Gravitational mass of the first neutron star [Msun]
        m2: Gravitational mass of the second neutron star [Msun]
        lambda_tilde: Effective tidal deformability (dimensionless)

    Returns:
        Disk mass [Msun]
    """
    # Fit parameters
    L0 = 245.0
    alpha = 0.097
    beta = 0.241

    # Symmetrized fits for mass ratio and tidal deformability
    l1 = (lambda_tilde / L0) ** alpha * (m2 / m1) ** beta
    l2 = (lambda_tilde / L0) ** alpha * (m1 / m2) ** beta

    # Symmetric combinations of l1 and l2
    x1 = 2.0 * ((1.0 + m2 / m1)**(-1.0) + l1**(-1.0) - 1.0)
    x2 = 2.0 * ((1.0 + m1 / m2)**(-1.0) + l2**(-1.0) - 1.0)

    # Clip to valid physical range [0, 1]
    x1 = xp.minimum(xp.maximum(x1, 0.0), 1.0)
    x2 = xp.minimum(xp.maximum(x2, 0.0), 1.0)


    # Disk mass calculation
    mass_disk = 0.25 * (2.0 + x2) * (x2 - 1.0) ** 2.0 * m2 + 0.25 * (2.0 + x1) * (x1 - 1.0) ** 2.0 * m1
    return xp.maximum(mass_disk,0.001)


@jit
def calculate_bns_disk_mass_nedora(m1, m2, lambda_tilde):
    """
    Calculate the disk mass for a BNS system using the Nedora et al. (2021) model.

    Parameters:
        m1: gravitational mass of the first neutron star [Msun]
        m2: gravitational mass of the second neutron star [Msun]
        lambda_tilde: effective tidal deformability (dimensionless)

    Returns:
        Disk mass [Msun]
    """
    # Coefficients from Nedora et al. 2021
    b0 = -1.85
    b1 = 2.59
    b2 = 7.07e-4
    b3 = -0.733
    b4 = -8.08e-4
    b5 = 2.75e-7

    # Ensure q >= 1 using JAX/NumPy compatible conditional
    q = xp.where(m1 > m2, m1 / m2, m2 / m1)

    # Polynomial fit from Nedora et al.
    log10_MDisk = (
        b0 + b1 * q + b2 * lambda_tilde +
        b3 * q**2 + b4 * lambda_tilde * q + b5 * lambda_tilde**2)

    # Set to 0 if result is negative
    return xp.maximum(log10_MDisk, 0.0)


@jit
def calculate_bns_dynamical_velocity_radice(c1, c2, mass_ratio, Q):
    """
    Calculate the dynamical ejecta velocity for a BNS system using the Radice et al. (2018) model.

    Parameters:
        c1: Compactness of the first neutron star
        c2: Compactness of the second neutron star
        q: Mass ratio (m1 / m2)
        mass_ratio: Inverse mass ratio (m2 / m1)

    Returns:
        Dynamical ejecta velocity (dimensionless, units of c)
    """
    # Fit parameters from Radice et al. (2018)
    alpha, beta, gamma = -0.287, 0.494, -3.00

    return  alpha * Q * (1.0 + gamma * c1) + alpha * mass_ratio * (1.0 + gamma * c2) + beta


@jit
def calculate_bns_dynamical_velocity_coughlin(c1, c2, mass_ratio, Q):
    """
    Calculate the dynamical ejecta velocity for a BNS system using the Coughlin et al. (2019) model.

    Parameters:
        c1: Compactness of the first neutron star
        c2: Compactness of the second neutron star
        q: Mass ratio (m1 / m2)
        mass_ratio: Inverse mass ratio (m2 / m1)

    Returns:
        Dynamical ejecta velocity (dimensionless, units of c)
    """
    a, b, c = -0.3090, 0.657, -1.879

    return a * (1.0 + c * c1) * Q + a * (1.0 + c * c2) * mass_ratio + b

@jit
def calculate_bns_dynamical_velocity_kruger(mass_ratio):
    """
    Calculate the dynamical ejecta velocity for a BNS system using the Coughlin et al. (2019) model.

    Parameters:
        mass_ratio: Mass ratio (m1 / m2)

    Returns:
        Dynamical ejecta velocity (dimensionless, units of c)
    """
    a, b= 0.0149, 0.1493

    return a * mass_ratio + b

@jit
def calculate_bns_q(v_ejecta_dyn):
    """
    Calculate the dynamical ejecta velocity for a BNS system using the Coughlin et al. (2019) model.

    Parameters:
        mass_ratio: Mass ratio (m1 / m2)

    Returns:
        Dynamical ejecta velocity (dimensionless, units of c)
    """
    a, b= 0.0149, 0.1493

    return (v_ejecta_dyn-b) / a
