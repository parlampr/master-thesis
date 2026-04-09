# units_conversion.py — Hardcoded CGS physical constants
# Replaces gemma.utils.units_conversion (no astropy dependency)

# --- Unit conversions ---
km_cgs = 1e5                             # 1 km in cm
day_cgs = 86400.0                        # 1 day in seconds
Mpc_in_cm = 3.085677581491367e24         # 1 Mpc in cm

# --- Physical Constants in CGS ---
c_cgs = 2.99792458e10                    # Speed of light in cm/s
G_cgs = 6.6743e-8                        # Gravitational constant cm^3 g^-1 s^-2
h_cgs = 6.62607015e-27                   # Planck constant erg·s
k_B_cgs = 1.380649e-16                   # Boltzmann constant erg/K
sigma_sb_cgs = 5.670374419e-5            # Stefan-Boltzmann erg cm^-2 s^-1 K^-4
M_sun_cgs = 1.98892e33                   # Solar mass in grams

# --- Derived Constants ---
c_km_s = c_cgs / km_cgs                  # Speed of light in km/s
u_cgs = day_cgs                          # Seconds in a day
Mpc = Mpc_in_cm                          # Alias for clarity
kB_cgs = k_B_cgs                         # Alias for naming consistency
sb_cgs = sigma_sb_cgs                    # Alias for naming consistency

# Conversion factor: (Angstrom^4 erg / (J cm^2 m^2)) -> (erg / (Angstrom s cm^2))
CONVERSION_FACTOR = 1e-43
