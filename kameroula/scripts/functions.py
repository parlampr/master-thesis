# functions.py — Ejecta physics functions
# Replaces gemma.ejecta.functions
# These operate on SCALARS (called with float inputs), so use plain Python math.

import math


def calculate_ns_compactness(lambda_ns, quarks, systematics_fraction=0.0):
    """
    Calculate the compactness of a neutron star from its tidal deformability.

    Parameters:
        lambda_ns: Dimensionless tidal deformability (must be > 0)
        quarks: Whether to include quark-matter corrections
        systematics_fraction: Fractional systematic uncertainty

    Returns:
        Compactness C = GM/Rc²
    """
    log_lambda = math.log(float(lambda_ns))

    val_no_quark = 0.371 - 0.0391 * log_lambda + 0.001056 * log_lambda ** 2.0
    val_quark = 0.360 - 0.0355 * log_lambda + 0.000705 * log_lambda ** 2.0

    compactness = val_quark if quarks else val_no_quark
    return compactness * (1.0 + systematics_fraction)


def calculate_baryonic_mass(mass, compactness):
    """
    Estimate the baryonic mass from gravitational mass and compactness.

    Parameters:
        mass: Gravitational mass [Msun]
        compactness: Dimensionless compactness

    Returns:
        Baryonic mass [Msun]
    """
    return mass + mass * (0.6 * compactness / (1.0 - 0.5 * compactness))


def calculate_r_isco(chi):
    """
    Radius of the innermost stable circular orbit (ISCO) around a Kerr BH.

    Parameters:
        chi: Dimensionless spin parameter

    Returns:
        ISCO radius in units of GM/c²
    """
    chi = float(chi)
    Z1 = 1.0 + (1.0 - chi**2) ** (1.0 / 3.0) * (
        (1.0 + chi) ** (1.0 / 3.0) + (1.0 - chi) ** (1.0 / 3.0))
    Z2 = math.sqrt(3.0 * chi**2.0 + Z1**2.0)
    sign_chi = 1.0 if chi >= 0 else -1.0
    return 3.0 + Z2 - sign_chi * math.sqrt(
        (3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2))


def calculate_orbital_energy(radius, chi):
    """Specific orbital energy in a circular orbit around a Kerr BH."""
    radius, chi = float(radius), float(chi)
    sign_chi = 1.0 if chi >= 0 else -1.0
    numerator = radius**2.0 - 2.0 * radius + sign_chi * math.sqrt(radius)
    denominator = radius * math.sqrt(
        radius**2.0 - 3.0 * radius + 2.0 * sign_chi * math.sqrt(radius))
    return numerator / denominator


def calculate_orbital_angular_momentum(radius, chi):
    """Specific orbital angular momentum in a circular orbit around a Kerr BH."""
    radius, chi = float(radius), float(chi)
    sign_chi = 1.0 if chi >= 0 else -1.0
    numerator = sign_chi * (
        radius**2.0 - 2.0 * sign_chi * math.sqrt(radius) + chi**2.0)
    denominator = math.sqrt(radius * (
        radius**2.0 - 3.0 * radius + 2.0 * sign_chi * math.sqrt(radius)))
    return numerator / denominator


def calculate_transition_factor(eta):
    """
    Transition factor based on symmetric mass ratio.

    Parameters:
        eta: Symmetric mass ratio

    Returns:
        Transition factor (0 to 1)
    """
    eta = float(eta)
    if eta <= 0.16:
        return 0.0
    if 0.16 < eta < 2.0 / 9.0:
        return 0.5 * (1.0 - math.cos(
            math.pi * (eta - 0.16) / (2.0 / 9.0 - 0.16)))
    if 2.0 / 9.0 <= eta <= 0.25:
        return 1.0
    return 0.0


def calculate_eff_tidal_deformability(mass_1, mass_2, lambda_1, lambda_2):
    """Effective tidal deformability for a BNS system."""
    return (16.0 / 13.0 * (
        (mass_1 + 12.0 * mass_2) * mass_1**4.0 * lambda_1 +
        (mass_2 + 12.0 * mass_1) * mass_2**4.0 * lambda_2
    ) / (mass_1 + mass_2) ** 5.0)


def calculate_energy_conversion_efficiency(chi_final):
    """Energy conversion efficiency based on final BH spin."""
    chi_final = float(chi_final)
    if chi_final <= 0.25:
        return 1e-4 * math.exp((chi_final - 0.25) / 0.06)
    elif chi_final <= 0.505:
        return 1e-4
    else:
        omega_H = chi_final / (1.0 + math.sqrt(1.0 - chi_final**2))
        return 0.068 * omega_H**5


def root_finding_final_bh_spin(final_bh_spin, chi_1, m1, m2, mb2, m_disk, eta, r_isco_init):
    """Root-finding equation for final BH spin after merger."""
    r_isco = calculate_r_isco(final_bh_spin)
    transition = calculate_transition_factor(eta)

    L_orb = calculate_orbital_angular_momentum(r_isco, final_bh_spin)
    E_orb_i = calculate_orbital_energy(r_isco_init, chi_1)
    E_orb_f = calculate_orbital_energy(r_isco, final_bh_spin)

    numerator = (chi_1 * m1**2 + L_orb * m1)
    numerator *= ((1.0 - transition) * m2 + transition * mb2 - m_disk)

    denominator = (
        (m1 + m2) * (1.0 - (1.0 - E_orb_i) * eta) -
        E_orb_f * m_disk
    ) ** 2

    return numerator / denominator - final_bh_spin
