# bns.py — BNS ejecta property calculations
# Replaces gemma.ejecta.bns — uses local scripts.backend
# Bilby mass conversion functions are inlined (no external dependency)

import jax.numpy as jnp
import numpy as np
# backend replaced by jax.numpy
from .functions import calculate_ns_compactness, calculate_baryonic_mass


# ── Bilby-equivalent mass conversions (inlined) ────────────────────────────

def _component_masses_to_chirp_mass(m1, m2):
    return (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)


def _chirp_mass_and_mass_ratio_to_component_masses(chirp_mass, mass_ratio):
    """mass_ratio = m2/m1 <= 1"""
    m1 = chirp_mass * (1.0 + mass_ratio) ** (1.0 / 5.0) / mass_ratio ** (3.0 / 5.0)
    m2 = mass_ratio * m1
    return m1, m2


def _lambda_1_lambda_2_to_lambda_tilde(lambda_1, lambda_2, mass_1, mass_2):
    mt = mass_1 + mass_2
    return (16.0 / 13.0) * (
        (mass_1 + 12.0 * mass_2) * mass_1**4 * lambda_1 +
        (mass_2 + 12.0 * mass_1) * mass_2**4 * lambda_2
    ) / mt**5


def _lambda_tilde_to_lambda_1_lambda_2(lambda_tilde, mass_1, mass_2):
    """Simple equal-lambda approximation: lambda_1 = lambda_2 = lambda_tilde."""
    # Full inversion requires an EOS; this symmetric approximation is standard
    # for cases where only lambda_tilde is given.
    return lambda_tilde, lambda_tilde


# ── Main BNS properties function ───────────────────────────────────────────

def bns_properties(mass_1=None, mass_2=None,
                   lambda_1=None, lambda_2=None,
                   chirp_mass=None, mass_ratio=None,
                   lambda_tilde=None,
                   fix_xsi=True, xsi_override=0.0,
                   m_tov=2.1, quarks=False,
                   systematics_fraction=0.0,
                   use_mosfit_model=False,
                   use_uncertainty=False,
                   err_v_dyn=0.01, err_m_disk=0.004,
                   formula_index=0,
                   key=None,
                   dynamical_mass=None,
                   dynamical_velocity=None,
                   disk_mass=None,
                   error_func=None,
                   preprocessed=False):
    """
    Compute BNS ejecta properties (disk mass, dynamical mass, wind mass, velocity).

    Returns:
        dict with keys: m_ejecta_dyn, m_ejecta_disk, m_ejecta_wind, v_ejecta_dyn,
                        m_tov, threshold_mass, compactness_1, compactness_2
    """
    if not preprocessed:
        if mass_1 is not None and mass_2 is not None:
            mass_ratio = mass_2 / mass_1
            chirp_mass = _component_masses_to_chirp_mass(mass_1, mass_2)
        elif mass_ratio is not None and chirp_mass is not None:
            mass_1, mass_2 = _chirp_mass_and_mass_ratio_to_component_masses(
                chirp_mass, mass_ratio)

        if lambda_tilde is not None:
            lambda_1, lambda_2 = _lambda_tilde_to_lambda_1_lambda_2(
                lambda_tilde, mass_1, mass_2)
        else:
            lambda_tilde = _lambda_1_lambda_2_to_lambda_tilde(
                lambda_1, lambda_2, mass_1, mass_2)

    Q = mass_1 / mass_2
    eta = mass_1 * mass_2 / (mass_1 + mass_2)**2

    c1 = calculate_ns_compactness(lambda_1, quarks, systematics_fraction)
    c2 = calculate_ns_compactness(lambda_2, quarks, systematics_fraction)
    mb1 = calculate_baryonic_mass(mass_1, c1)
    mb2 = calculate_baryonic_mass(mass_2, c2)

    approx_radius_ns = jnp.where(
        use_mosfit_model,
        11.2 * chirp_mass * (lambda_tilde / 800)**(1 / 6),
        chirp_mass * (lambda_tilde / 0.0042)**(1 / 6))

    total_mass = mass_1 + mass_2
    threshold_mass = (2.38 - 3.606 * (m_tov / approx_radius_ns)) * m_tov

    xsi = jnp.where(
        fix_xsi,
        0.18 + (0.29 - 0.18) / (1.0 + jnp.exp(jnp.asarray(1.5 * (mass_1 / mass_2 - 3.0)))),
        xsi_override)

    # Dynamical ejecta mass
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
        raise ValueError(f"Unsupported dynamical_mass: {dynamical_mass}")

    # Dynamical velocity
    if dynamical_velocity == "radice":
        v_dyn = calculate_bns_dynamical_velocity_radice(c1, c2, mass_ratio, Q)
    elif dynamical_velocity == "coughlin":
        v_dyn = calculate_bns_dynamical_velocity_coughlin(c1, c2, mass_ratio, Q)
    else:
        raise ValueError(f"Unsupported dynamical_velocity: {dynamical_velocity}")

    # Disk mass
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
        raise ValueError(f"Unsupported disk_mass: {disk_mass}")

    if use_uncertainty:
        assert key is not None and error_func is not None
        sampled = error_func(
            m_dyn, v_dyn, m_disk, Q,
            err_v_dyn, err_m_disk,
            formula_index,
            not fix_xsi, xsi_override,
            key)
        m_dyn = sampled["m_dyn"]
        v_dyn = sampled["v_dyn"]
        m_disk = sampled["mdisk_low"]
        m_wind = sampled["m_wind"]
    else:
        m_wind = xsi * m_disk

    return {
        "m_ejecta_dyn": m_dyn, "m_ejecta_disk": m_disk,
        "m_ejecta_wind": m_wind, "v_ejecta_dyn": v_dyn,
        "m_tov": m_tov, "threshold_mass": threshold_mass,
        "compactness_1": c1, "compactness_2": c2
    }


# ── Dynamical mass models ─────────────────────────────────────────────────

def calculate_bns_dynamical_mass_kruger(m1, m2, c1, c2, mass_ratio, q):
    """Kruger et al. (2020) dynamical ejecta mass."""
    a, b, c, n = -9.33347796, 114.16916699, -337.55577457, 1.54648799
    term1 = (a / float(c1) + b * float(mass_ratio)**n + c * float(c1)) * float(m1)
    term2 = (a / float(c2) + b * float(q)**n + c * float(c2)) * float(m2)
    m_dyn = term1 + term2
    return max(m_dyn * 0.001, 1e-6)


def calculate_bns_dynamical_mass_coughlin(m1, m2, c1, c2):
    """Coughlin et al. (2019) dynamical ejecta mass."""
    a, b, d, n = -0.0719, 0.2116, -2.42, -2.905
    log_m_dyn = (
        a * (1.0 - 2.0 * c1) / c1 * m1 +
        b * m2 * (m1 / m2) ** n +
        a * (1.0 - 2.0 * c2) / c2 * m2 +
        b * m1 * (m2 / m1) ** n + d)
    return 10.0 ** log_m_dyn


def calculate_bns_dynamical_mass_radice(mass_1, mass_2, compactness_1, compactness_2,
                                         m_baryonic_1, m_baryonic_2):
    """Radice et al. (2018) dynamical ejecta mass."""
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
    return jnp.maximum(0.001 * m_dyn, 0.0)


def calculate_bns_dynamical_mass_dietrich(m1, m2, c1, c2, mb1, mb2):
    """Dietrich et al. (2016) dynamical ejecta mass."""
    a, b, c, d, n = -1.35695, 6.11252, -49.43355, 16.1144, -2.5484
    term1 = (a * (m2 / m1) ** (1.0 / 3.0) * (1.0 - 2.0 * c1) / c1 + b * (m2 / m1) ** n) * mb1
    term2 = (a * (m1 / m2) ** (1.0 / 3.0) * (1.0 - 2.0 * c2) / c2 + b * (m1 / m2) ** n) * mb2
    term3 = c * 1.2 * c1 * m1 / (2.0 - c1)
    term4 = c * 1.2 * c2 * m2 / (2.0 - c2)
    return 0.001 * jnp.maximum(term1 + term2 + term3 + term4 + d, 1.0)


def calculate_bns_dynamical_mass_nedora(m1, m2, lambda_tilde):
    """Nedora et al. (2021) dynamical ejecta mass."""
    b0, b1, b2, b3, b4, b5 = -1.32, -3.82e-1, -4.47e-3, -3.39e-1, 3.21e-3, 4.31e-7
    q = jnp.where(m1 > m2, m1 / m2, m2 / m1)
    log10_mdyn = (b0 + b1 * q + b2 * lambda_tilde +
                  b3 * q**2 + b4 * lambda_tilde * q + b5 * lambda_tilde**2)
    return 10.0 ** log10_mdyn


# ── Disk mass models ──────────────────────────────────────────────────────

def calculate_bns_disk_mass_kruger(m2, c2):
    """Kruger et al. (2020) disk mass."""
    a, c, d = -8.1324, 1.4820, 1.7784
    fit_base = a * float(c2) + c
    base = max(fit_base, 5e-4)
    return float(m2) * base**d


def calculate_bns_disk_mass_radice(lambda_tilde):
    """Radice et al. (2018) disk mass."""
    alpha, beta, gamma, delta = 0.084, 0.127, 567.1, 405.14
    m_disk = alpha + beta * jnp.tanh((lambda_tilde - gamma) / delta)
    return jnp.maximum(m_disk, 0.001)


def calculate_bns_disk_mass_coughlin(total_mass, threshold_mass):
    """Coughlin et al. (2019) disk mass."""
    a, b, c, d = -31.335, -0.976, 1.0474, 0.05957
    remnant_mass_log10 = a * (1 + b * jnp.tanh((c - total_mass / threshold_mass) / d))
    remnant_mass_log10 = jnp.maximum(remnant_mass_log10, -3.0)
    return 10.0 ** remnant_mass_log10


def calculate_bns_disk_mass_dietrich(mass_ratio, total_mass, threshold_mass):
    """Dietrich et al. (2020) disk mass."""
    beta_val = 3.910
    qtrans = 0.9
    xsi = 0.5 * jnp.tanh(beta_val * (mass_ratio - qtrans))
    a0, da, b0, db, c, d = -1.581, -2.439, -0.538, -0.406, 0.953, 0.0417
    remnant_mass_log10 = (a0 + da * xsi) * (
        1 + (b0 + db * xsi) * jnp.tanh((c - total_mass / threshold_mass) / d))
    remnant_mass_log10 = jnp.maximum(remnant_mass_log10, -3.0)
    return 10.0 ** remnant_mass_log10


def calculate_bns_disk_mass_barbieri(m1, m2, lambda_tilde):
    """Barbieri et al. (2021) disk mass."""
    L0, alpha, beta = 245.0, 0.097, 0.241
    l1 = (lambda_tilde / L0) ** alpha * (m2 / m1) ** beta
    l2 = (lambda_tilde / L0) ** alpha * (m1 / m2) ** beta
    x1 = 2.0 * ((1.0 + m2 / m1)**(-1.0) + l1**(-1.0) - 1.0)
    x2 = 2.0 * ((1.0 + m1 / m2)**(-1.0) + l2**(-1.0) - 1.0)
    x1 = jnp.minimum(jnp.maximum(x1, 0.0), 1.0)
    x2 = jnp.minimum(jnp.maximum(x2, 0.0), 1.0)
    mass_disk = (0.25 * (2.0 + x2) * (x2 - 1.0)**2.0 * m2 +
                 0.25 * (2.0 + x1) * (x1 - 1.0)**2.0 * m1)
    return jnp.maximum(mass_disk, 0.001)


def calculate_bns_disk_mass_nedora(m1, m2, lambda_tilde):
    """Nedora et al. (2021) disk mass."""
    b0, b1, b2, b3, b4, b5 = -1.85, 2.59, 7.07e-4, -0.733, -8.08e-4, 2.75e-7
    q = jnp.where(m1 > m2, m1 / m2, m2 / m1)
    log10_MDisk = (b0 + b1 * q + b2 * lambda_tilde +
                   b3 * q**2 + b4 * lambda_tilde * q + b5 * lambda_tilde**2)
    return jnp.maximum(log10_MDisk, 0.0)


# ── Dynamical velocity models ─────────────────────────────────────────────

def calculate_bns_dynamical_velocity_radice(c1, c2, mass_ratio, Q):
    """Radice et al. (2018) dynamical ejecta velocity."""
    alpha, beta, gamma = -0.287, 0.494, -3.00
    return alpha * Q * (1.0 + gamma * c1) + alpha * mass_ratio * (1.0 + gamma * c2) + beta


def calculate_bns_dynamical_velocity_coughlin(c1, c2, mass_ratio, Q):
    """Coughlin et al. (2019) dynamical ejecta velocity."""
    a, b, c = -0.3090, 0.657, -1.879
    return a * (1.0 + c * c1) * Q + a * (1.0 + c * c2) * mass_ratio + b
