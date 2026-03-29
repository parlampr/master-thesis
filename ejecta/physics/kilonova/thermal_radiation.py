from gemma.backend import get_arraylib, jit_if_available
from math import pi

xp = get_arraylib()
jit = jit_if_available()

# Constants
h_cgs = 6.62607015e-27
c_cgs = 2.99792458e10
k_B_cgs = 1.380649e-16

@jit
def blackbody_nu(T, nu):
    """
    Planck function B_nu (erg/s/cm²/Hz/sr).
    """
    exponent = h_cgs * nu / (k_B_cgs * T)
    # Use expm1 for numerical safety
    return (2.0 * h_cgs * nu**3 / c_cgs**2) / xp.expm1(exponent)

@jit
def blackbody_lam(T, lam):
    """
    Planck function B_lambda (erg/s/cm²/cm/sr).
    """
    exponent = h_cgs * c_cgs / (lam * k_B_cgs * T)
    return (2.0 * h_cgs * c_cgs**2 / lam**5) / xp.expm1(exponent)

@jit
def heating_rate_korobkin(t, eth=0.5):
    """
    Radioactive heating rate from Korobkin et al. (2012).
    """
    eps0 = 2e18
    t0 = 1.3
    sigma = 0.11
    alpha = 1.3

    bracket = 0.5 - (1.0 / pi) * xp.arctan((t - t0) / sigma)
    return eps0 * bracket**alpha * eth / 0.5
