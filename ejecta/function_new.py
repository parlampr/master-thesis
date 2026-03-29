from gemma.backend import get_arraylib, jit_if_available
from astropy.constants import c

xp = get_arraylib()
jit = jit_if_available()


@jit
def calculate_ns_compactness(lambda_ns, quarks, systematics_fraction=0.0):
    """
    Calculate the compactness of a neutron star based on its dimensionless tidal deformability (lambda),
    including optional systematic fractional correction.

    Parameters:
        * `lambda_ns` (float or array-like): Dimensionless tidal deformability of the neutron star. Must be > 0.
        * `quarks` (bool or array-like): Whether to include quark-matter corrections in the compactness estimate.
        * `systematics_fraction` (float): Fractional systematic uncertainty to add to the compactness values (e.g., 0.05 for +5%).

    Returns:
        * Compactness (float or array-like): Dimensionless compactness C = GM/Rc² of the neutron star.

    Notes:
        Uses a backend-compatible conditional via `xp.where()` to avoid JAX TracerBoolConversionError.
        `systematics_fraction` is applied multiplicatively.
    """
    log_lambda = xp.log(lambda_ns)

    val_no_quark = 0.371 - 0.0391 * log_lambda + 0.001056 * log_lambda ** 2.0
    val_quark = 0.360 - 0.0355 * log_lambda + 0.000705 * log_lambda ** 2.0

    compactness = xp.where(quarks, val_quark, val_no_quark)
    #print("Yves",compactness * (1.0 + systematics_fraction))
    return compactness * (1.0 + systematics_fraction)

@jit
def calculate_disk_ejecta(mass_ratio, xi1=0.18, xi2=0.29, m_disk):
    xi = xi1 + (xi2-xi1)/(1+xp.exp(1.5*(mass_ratio-3)))

    return m_disk*xi

@jit
def calculate_accretion_mass(mass_ratio, xi1=0.18, xi2=0.29, m_disk):
    return (m_disk - calculate_disk_ejecta(mass_ratio, xi1, xi2))

@jit
def calculate_accretion_mass2(m_disk, xsi):
    return (m_disk * (1 - xsi))

@jit
def calculate_accretion_mass3(E_iso, theta_c, remnant_spin,frac=0.1):
    return jet_kinetic2(E_iso, theta_c) / (0.5*(1-frac)*accretion_efficiency(remnant_spin)*c**2)

@jit
def accretion_efficiency(remnant_spin):
    return 1.52*xp.exp(remnant_spin/0.06)*1e-6

@jit
def jet_kinetic(frac=0.1, remnant_spin, m_disk, xsi):
    return 0.5*(1-frac)*accretion_efficiency(remnant_spin)*calculate_accretion_mass2(m_disk,xsi)*c**2

@jit
def jet_kinetic2(E_iso, theta_c):
    return E_iso*(1-theta_c)

    
@jit
def calculate_baryonic_mass(mass, compactness):
    """
    Estimate the baryonic mass of a neutron star from its gravitational mass and compactness.

    Parameters:
        * `mass` (float): Gravitational mass of the neutron star (solar masses).
        * `compactness` (float): Compactness of the neutron star (dimensionless).

    Returns:
        * Baryonic mass of the neutron star (solar masses).
    """

    return mass + mass * (0.6 * compactness / (1.0 - 0.5 * compactness))

@jit
def calculate_r_isco(chi):
    """
    Calculate the radius of the innermost stable circular orbit (ISCO) around a Kerr black hole.

    Parameters:
        * `chi` (float): Dimensionless spin parameter of the black hole.

    Returns:
        * ISCO radius in units of GM/c^2.
    """

    Z1 = 1.0 + (1.0 - chi**2) ** (1.0 / 3.0) * ((1.0 + chi) ** (1.0 / 3.0) + (1.0 - chi) ** (1.0 / 3.0))
    Z2 = xp.sqrt(3.0 * chi**2.0 + Z1**2.0)
    return 3.0 + Z2 - xp.sign(chi) * xp.sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2))

@jit
def calculate_orbital_energy(radius, chi):
    """
    Calculate the specific orbital energy of a test particle  in a circular orbit around
    a Kerr black hole.

    Parameters:
        * `radius` (float): Orbital radius in units of GM/c^2.
        * `chi` (float): Dimensionless spin parameter of the black hole.

    Returns:
        *  Specific orbital energy.
    """

    numerator = radius**2.0 - 2.0 * radius + xp.sign(chi) * xp.sqrt(radius)
    denominator = radius * xp.sqrt(radius**2.0 - 3.0 * radius + 2.0 * xp.sign(chi) * xp.sqrt(radius))
    return numerator / denominator

@jit
def calculate_orbital_angular_momentum(radius, chi):
    """
    Calculate the specific orbital angular momentum of a test particle in a circular orbit around
    a Kerr black hole.

    Parameters:
        * `radius` (float): Orbital radius in units of GM/c^2.
        * `chi` (float): Dimensionless spin parameter of the black hole.

    Returns:
        * Specific orbital angular momentum.
    """

    numerator = xp.sign(chi) * (radius**2.0 - 2.0 * xp.sign(chi) * xp.sqrt(radius) + chi**2.0)
    denominator = xp.sqrt(radius * (radius**2.0 - 3.0 * radius + 2.0 * xp.sign(chi) * xp.sqrt(radius)))
    return numerator / denominator

@jit
def calculate_transition_factor(eta):
    """
    Calculate the transition factor based on the symmetric mass ratio.

    Parameters:
        * `eta` (float): Symmetric mass ratio.

    Returns:
        * Transition factor.
    """

    if eta <= 0.16:
        return 0.0
    if 0.16 < eta < 2.0 / 9.0:
        return 0.5 * (1.0 - xp.cos(xp.pi * (eta - 0.16) / (2.0 / 9.0 - 0.16)))
    if 2.0 / 9.0 <= eta <= 0.25:
        return 1.0

@jit
def calculate_eff_tidal_deformability(mass_1, mass_2, lambda_1, lambda_2):
    """
    Calculate the effective tidal deformability for a BNS system.

    Parameters:
        * `mass_1` (float): Mass of the heaviest neutron star (solar masses).
        * `mass_2` (float): Mass of the lightest neutron star (solar masses).
        * `lambda_1` (float): Tidal deformability of the heaviest neutron star.
        * `lambda_2` (float): Tidal deformability of the lightest neutron star.

    Returns:
        * Effective tidal deformability of the system.
    """

    return (16.0 / 13.0 * ((mass_1 + 12.0 * mass_2) * mass_1**4.0 * lambda_1 + (mass_2 + 12.0 * mass_1) * mass_2**4.0 * lambda_2)
        / (mass_1 + mass_2) ** 5.0)



@jit
def calculate_energy_conversion_efficiency(chi_final):
    """
    Calculate the energy conversion efficiency based on the final black hole spin.

    Parameters:
        chi_final: Final spin of the black hole (dimensionless)

    Returns:
        Energy conversion efficiency (dimensionless fraction)
    """
    # Case 1: low-spin regime (smooth exponential rise)
    if chi_final <= 0.25:
        return 1e-4 * xp.exp((chi_final - 0.25) / 0.06)

    # Case 2: intermediate regime (plateau)
    elif chi_final <= 0.505:
        return 1e-4

    # Case 3: high-spin regime (ergosphere contribution)
    omega_H = chi_final / (1.0 + xp.sqrt(1.0 - chi_final**2))
    return 0.068 * omega_H**5

def calculate_final_bh_spin(self):
    """
    Calculate the final black hole spin after a binary merger.

    Returns:
        Final BH spin (dimensionless)
    """
    def root_fn(spin):
        return root_finding_final_bh_spin(
            spin,
            self.parameters["chi_1"],
            self.parameters["mass_1"],
            self.parameters["mass_2"],
            self.m_baryonic_2,
            self.m_disk,
            self.eta,
            self.r_isco)

    return root_scalar_flexible(root_fn, bracket=(0.01, 0.99))


@jit
def root_finding_final_bh_spin(final_bh_spin, chi_1, m1, m2, mb2, m_disk, eta, r_isco_init):
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
