# hotokezaka_lightcurve.py
from gemma.backend import get_arraylib, jit_if_available, get_backend_name, get_erfc
from gemma.utils.units_conversion import*
from .lightcurve_base import EMbase
import pkg_resources
import pyphot
import extinction
from astropy import units as u
from astropy import constants as const
from numba.typed import List
import numpy as np
from math import pi, exp, log, pow

# Backend setup
xp = get_arraylib()
jit = jit_if_available()
backend_name = get_backend_name()

# Handle erfc switching
if backend_name == "jax":
    from jax import lax
    erfc = get_erfc()
    def _calc_lightcurve(*args, **kwargs):
        return _calc_lightcurve_jax(*args, **kwargs)
else:
    from math import erfc as math_erfc
    from numba import vectorize, float64, prange
    @vectorize([float64(float64)])
    def erfc(x):
        return math_erfc(x)

    def _calc_lightcurve(*args, **kwargs):
        return _calc_lightcurve_numba(*args, **kwargs)


NSBH_COMPONENTS = ['dynamic', 'wind', 'magnetic']
BNS_COMPONENTS = ['dynamic', 'wind', 'magnetic']


class Hotokezaka(EMbase):
    def __init__(self, components=['dynamic'], sample_gw_parameters=False,
                                               gw_param_mode="chirp_mass",
                                               ejecta_function=None,
                                                use_extinction=False):



        self.model_wavelengths = xp.linspace(100.0, 99900.0, 500)  # Angstroms
        self.model_wavelengths_cgs = (self.model_wavelengths * u.AA).to(u.cm).value  # In cm

        if backend_name == "jax":
            # To ensure compatibility and avoid precision drift when using JAX,
            # convert the NumPy-computed CGS wavelengths explicitly to jax.numpy.
            # This guarantees identical inputs to spectral calculations across backends.

            self.model_wavelengths_cgs = xp.array(self.model_wavelengths_cgs)

        self.components = components
        self.use_extinction = use_extinction
        self.sample_gw_parameters = sample_gw_parameters
        self.ejecta_function = ejecta_function
        self.gw_param_mode = gw_param_mode

        super().__init__(hotokezaka_model=True)

        # --- Dynamic parameter setup ---
        self.parameter_names = []
        self.parameter_bounds = {}

        component_suffix_map = {"dynamic": "dyn", "wind": "wind","magnetic": "magnetic"}

        # Shared parameter structure
        base_params = [
            ("m_ejecta", (1e-4, 0.2)),     # M_sun
            ("v_ejecta", (0.05, 0.4)),     # fraction of c
            ("alpha_max", (1.0, 5.0)),
            ("alphas_min", (0.01, 2.0)),
            ("n", (2.0, 10.0)),
            ("kappa_low", (0.1, 1.0)),     # cm^2/g
            ("kappa_high", (1.0, 20.0)),   # cm^2/g
        ]

        for comp in self.components:
            suffix = component_suffix_map[comp]
            for base_name, bounds in base_params:
                pname = f"{base_name}_{suffix}"
                self.parameter_names.append(pname)
                self.parameter_bounds[pname] = bounds

        # Magnetic-only parameters
        if "magnetic" in self.components:
            self.parameter_names.append("fraction_magnetic")
            self.parameter_bounds["fraction_magnetic"] = (0.01, 1.0)

            self.parameter_names.append("m_disk")
            self.parameter_bounds["m_disk"] = (0.01, 0.3)

        # Add lightcurve integration and distance parameters
        self.parameter_names += ["t_max", "dt", "luminosity_distance"]
        self.parameter_bounds["t_max"] = (1.0, 30.0)         # days
        self.parameter_bounds["dt"] = (0.01, 2.0)            # days
        self.parameter_bounds["luminosity_distance"] = (0.1, 1000.)            # cm (~0.1–1000 Mpc)


    def temperature_and_radius(self):
        m_ejectas, v_ejectas, alphas_max, alphas_min, indices_n, kappas_low, kappas_high, be_kappas  = [], [], [], [], [], [], [], []

        # dynamical ejecta component
        if 'dynamic' in self.components:
            m_ejectas = m_ejectas + [self.params["m_ejecta_dyn"]]
            v_ejectas = v_ejectas + [self.params["v_ejecta_dyn"]]
            alphas_max = alphas_max + [self.params["alpha_max_dyn"]]
            alphas_min = alphas_min + [self.params["alphas_min_dyn"]]
            indices_n = indices_n + [self.params["n_dyn"]]
            kappas_low = kappas_low + [self.params["kappa_low_dyn"]]
            kappas_high = kappas_high + [self.params["kappa_high_dyn"]]
            be_kappas = be_kappas + [self.params["be_kappa_dyn"]]


        # wind ejecta component
        if 'wind' in self.components:
            m_ejectas = m_ejectas + [self.params["m_ejecta_wind"]]
            v_ejectas = v_ejectas + [self.params["v_ejecta_wind"]]
            alphas_max = alphas_max + [self.params["alpha_max_wind"]]
            alphas_min = alphas_min + [self.params["alphas_min_wind"]]
            indices_n = indices_n + [self.params["n_wind"]]
            kappas_low = kappas_low + [self.params["kappa_low_wind"]]
            kappas_high = kappas_high + [self.params["kappa_high_wind"]]
            be_kappas = be_kappas + [self.params["be_kappa_wind"]]

        # magnetic component
        if 'magnetic' in self.components:
            m_ejectas = m_ejectas + [self.params["fraction_magnetic"] * self.params["m_disk"]]
            v_ejectas = v_ejectas + [self.params["v_ejecta_magnetic"]]
            alphas_max = alphas_max + [self.params["alpha_max_magnetic"]]
            alphas_min = alphas_min + [self.params["alphas_min_magnetic"]]
            indices_n = indices_n + [self.params["n_magnetic"]]
            kappas_low = kappas_low + [self.params["kappa_low_magnetic"]]
            kappas_high = kappas_high + [self.params["kappa_high_magnetic"]]
            be_kappas = be_kappas + [self.params["be_kappa_magnetic"]]

        self.TemperaturePhotosphere, self.RadiusPhotosphere = [], []

        # compute photosphere radius and temperature for each component
        for i, (m_ejecta, v_ejecta, alpha_max, alpha_min, n, kappa_low, kappa_high, be_kappa) in enumerate(zip(m_ejectas, v_ejectas, alphas_max, alphas_min, indices_n, kappas_low, kappas_high, be_kappas)):
            be_array, Nbeta, be_min, be_max,dbe = get_be_array(v_ejecta * c_cgs, alpha_min, alpha_max, Nbeta=100)
            t_array = xp.arange(0.01 * u_cgs, self.params['t_max'] * u_cgs + self.params['dt']*u_cgs, self.params['dt']*u_cgs)
            self._t, luminosities, temperature_photosphere, radius_photosphere = _calc_lightcurve(m_ejecta * M_sun_cgs, v_ejecta * c_cgs, alpha_max, alpha_min, n, kappa_low, kappa_high, be_kappa,
                                                                                                   be_array, Nbeta, be_min, be_max,dbe,t_array)
            if i > 0:
                if len(self._t) > len(dummy_t):
                    while len(self._t) > len(dummy_t):
                        self._t = self._t[:-1]
                        temperature_photosphere = temperature_photosphere[:-1]
                        radius_photosphere = radius_photosphere[:-1]

                elif len(self._t) < len(dummy_t):
                    while len(self._t) < len(dummy_t):
                        self._t = xp.append(self._t, self._t[-1])
                        temperature_photosphere = xp.append(temperature_photosphere, temperature_photosphere[-1])
                        radius_photosphere = xp.append(radius_photosphere, radius_photosphere[-1])

            self.TemperaturePhotosphere.append(temperature_photosphere)
            self.RadiusPhotosphere.append(radius_photosphere)
            dummy_t = self._t



    def generate_light(self, params):
        self.params = params

        self.temperature_and_radius()

    def generate_spectra(self):
        """
        Generate SEDs for all time steps and ejecta components.
        Returns:
        - spectra: array
            Spectral flux density (erg/s/cm^2/Å) for each time step, combining all components.
        """
        temps = xp.asarray(self.TemperaturePhotosphere)  # ensure array
        radii = xp.asarray(self.RadiusPhotosphere)

        #print("temps", radii)
        return generate_spectra_jit(temps, radii, self.model_wavelengths_cgs, self.D)

@jit
def generate_spectra_jit(temperature, radius, model_wavelengths, distance):
    """
    Generate SEDs for all time steps and ejecta components.

    Returns:
    - spectra: array
        Spectral flux density (erg/s/cm^2/Å) for each time step, combining all components.
    """
    #temps = xp.stack(temperature)  # (n_components, n_times)
    #radii = xp.stack(radius)        # (n_components, n_times)



    # Broadcast temp and wavelength
    bb_flux = blackbody_lam(temperature[..., None], model_wavelengths) * pi  # (n_components, n_times, n_wavelengths)

    # Scale by radius and distance
    flux = bb_flux * (radius[..., None]**2) / (distance**2)
    bb_flux = bb_flux/CONVERSION_FACTOR



    # Sum over components
    spectra = xp.sum(flux, axis=0)  # (n_times, n_wavelengths)
    #print(spectra)
    return spectra

def get_be_array(vej, alpha_min, alpha_max, Nbeta):
    be_min = vej * alpha_min / c_cgs
    be_max = vej * alpha_max / c_cgs
    dbe = (be_max - be_min) / Nbeta
    return xp.arange(be_min, be_max, dbe), Nbeta, be_min, be_max,dbe


# -------------------
# JITted Functions
# -------------------

@jit
def blackbody_lam(temperature, lam):
    exp_arg = h_cgs * c_cgs / (lam * kB_cgs * temperature)
    return (2.0 * h_cgs * c_cgs**2 / (lam**5 * (xp.exp(exp_arg) - 1.0))) * 1e-8 # u.erg/u.s/u.cm**2/u.AA



@jit
def _heating_rate(t, eth=0.5):
    eps0 = 2e18  # erg/g/s
    t0 = 1.3     # s
    sig = 0.11   # s
    alpha = 1.3  # dimensionless
    brac = 0.5 - 1.0 / pi * xp.arctan((t - t0) / sig)
    return eps0 * brac**alpha * eth / 0.5


@jit
def compute_ymax(t_dif, t_RK):
    arg = 0.5 * t_dif / t_RK
    arg = xp.where(arg > 0, arg, 0.0)
    return xp.sqrt(arg)

@jit
def compute_luminosity(E, t_RK, t_dif, beta, heat, dt):
    tesc = xp.minimum(t_dif, t_RK) + beta * t_RK
    ymax = compute_ymax(t_dif, t_RK)
    erfc_vals = erfc(ymax)  # fast + vectorized

    L = erfc_vals * E / tesc
    dE = (-E / t_RK - L + heat) * dt
    return L, dE

@jit
def _interp_logx_logy(x, x_arr, y_arr):
    for i in range(len(x_arr) - 1):
        if x_arr[i] <= x <= x_arr[i+1]:
            x0, x1 = x_arr[i], x_arr[i+1]
            y0, y1 = y_arr[i], y_arr[i+1]
            return exp(log(y0) + (log(y1) - log(y0)) * (log(x) - log(x0)) / (log(x1) - log(x0)))
    return y_arr[-1]  # fallback


@jit
def log_interp(x, xp, fp):
    for i in range(len(xp) - 1):
        if xp[i] <= x <= xp[i + 1]:
            slope = (fp[i + 1] - fp[i]) / (xp[i + 1] - xp[i])
            return fp[i] + slope * (x - xp[i])
    return fp[-1]

@jit
def _calc_lightcurve_numba(Mej, vej,
                    alpha_max, alpha_min, n,
                    kappa_low, kappa_high,
                    be_kappa, be_array, Nbeta, be_min, be_max, dbe, t_array):

    rho0 = Mej*(n-3.)/(4.*pi*pow(vej,3.))/(1.-pow(alpha_max/alpha_min,-n+3.))

    ind_low = np.where(be_array <= be_kappa)[0]
    ind_high = np.where(be_array > be_kappa)[0]
    tau_low = kappa_low*be_min*c_cgs*rho0*(pow(be_kappa/be_min,-n+1.)-pow(be_max/be_min,-n+1.))/(n-1.)+kappa_high*be_min*c_cgs*rho0*((be_array[ind_low]/be_min)**(-n+1.)-pow(be_kappa/be_min,-n+1.))/(n-1.)
    tau_high = kappa_low*be_min*c_cgs*rho0*((be_array[ind_high]/be_min)**(-n+1.)-pow(be_max/be_min,-n+1.))/(n-1.)

    dMs = 4.*pi*pow(vej,3.)*rho0*(be_array/be_min)**(-n+2.)*dbe/be_min
    taus = np.concatenate((tau_low, tau_high))
    sortidx = np.argsort(taus)
    bes = be_array
    tds = taus*bes
    Eins = np.zeros((len(bes)))

    dt = t_array[1]-t_array[0]
    t = 0.01*u_cgs
    ts = [] #np.array([0.])
    Ls = [] #np.array([0.])
    temps = [] #np.array([0.])
    Rph = [] #np.array([0.])
    j = 0
    k= 0
    for t in t_array:
        heat_th0 = _heating_rate(t)
        heat_th1 = _heating_rate(t + 0.5 * dt)
        heat_th2 = _heating_rate(t + dt)

        # RK1
        t_RK1 = t
        t_dif1 = tds / t_RK1
        heat1 = dMs * heat_th0
        L1, dE1 = compute_luminosity(Eins, t_RK1, t_dif1, bes, heat1, dt)

        # RK2
        E2 = Eins + 0.5 * dE1
        t_RK2 = t + 0.5 * dt
        t_dif2 = tds / t_RK2
        heat2 = dMs * heat_th1
        L2, dE2 = compute_luminosity(E2, t_RK2, t_dif2, bes, heat2, dt)

        # RK3
        E3 = Eins + 0.5 * dE2
        t_RK3 = t + 0.5 * dt
        t_dif3 = tds / t_RK3
        heat3 = dMs * heat_th1
        L3, dE3 = compute_luminosity(E3, t_RK3, t_dif3, bes, heat3, dt)

        # RK4
        E4 = Eins + dE3
        t_RK4 = t + dt
        t_dif4 = tds / t_RK4
        heat4 = dMs * heat_th2
        L4, dE4 = compute_luminosity(E4, t_RK4, t_dif4, bes, heat4, dt)

        # Final update
        Eins += (dE1 + 2*dE2 + 2*dE3 + dE4) / 6.0
        Ltot = np.sum((L1 + 2*L2 + 2*L3 + L4) / 6.0)

        t += dt
        #print(t, Ltot)
        #search for the shell of tau = 1

        if(taus[0]/(t*t)>1. and taus[len(bes)-1]/(t*t)<1.):
#             print(np.log(t*t), np.log(taus), np.log(bes))
            be = np.exp(np.interp(np.log(t*t), np.log(taus[sortidx]), np.log(bes[sortidx])))
            r = be*c_cgs*t
            Eint = np.exp(np.interp(np.log(t*t), np.log(taus[sortidx]), np.log(Eins[sortidx])))
        elif(taus[len(bes)-1]/(t*t)>1.):
            l = len(bes)-1
            be = bes[l]
            r = be*c_cgs*t
            Eint = Eins[l]
        else:
            l = 0
            be = bes[0]
            r = be*c_cgs*t
            Eint = Eins[0]

        tmp = Ltot/(4.*pi*sb_cgs*r*r);
        if tmp==0.0 or Ltot<0.0:
            continue
        temp = pow(tmp,0.25)


        if(j<10):
            Ls.append(Ltot)
            ts.append(t)
            temps.append(temp)
            Rph.append(r)

        elif(j<100):
            if(j%3==0):
                 Ls.append(Ltot)
                 ts.append(t)
                 temps.append(temp)
                 Rph.append(r)

        elif(j<1000):
            if(j%30==0):
                 Ls.append(Ltot)
                 ts.append(t)
                 temps.append(temp)
                 Rph.append(r)

        elif(j<10000):
            if(j%100==0):
                 Ls.append(Ltot)
                 ts.append(t)
                 temps.append(temp)
                 Rph.append(r)


        j += 1
    # Changing to np.array
    Ls = np.array(Ls)
    ts = np.array(ts)
    temps = np.array(temps)
    Rph = np.array(Rph)

    return ts/u_cgs, Ls, temps, Rph

def jax_interp(x, p, fp):
    """Linear 1D interpolation compatible with JAX JIT."""
    idx = xp.clip(xp.searchsorted(p, x) - 1, 0, len(p) - 2)
    x0 = p[idx]
    x1 = p[idx + 1]
    y0 = fp[idx]
    y1 = fp[idx + 1]
    slope = (y1 - y0) / (x1 - x0)
    return y0 + slope * (x - x0)

@jit
def thinning_mask(t_array):
    idxs = xp.arange(t_array.shape[0])
    cond1 = idxs < 10
    cond2 = (idxs < 100) & (idxs % 3 == 0)
    cond3 = (idxs < 1000) & (idxs % 30 == 0)
    cond4 = (idxs < 10000) & (idxs % 100 == 0)
    return cond1 | cond2 | cond3 | cond4

@jit
def _calc_lightcurve_jax_raw(Mej, vej,
                    alpha_max, alpha_min, n,
                    kappa_low, kappa_high,
                    be_kappa, be_array, Nbeta, be_min, be_max, dbe, t_array):

    rho0 = Mej * (n - 3.0) / (4.0 * pi * vej**3) / (1.0 - (alpha_max / alpha_min)**(-n + 3.0))

    mask_low = be_array <= be_kappa

    tau_low = (
        kappa_low * be_min * c_cgs * rho0 *
        ((be_kappa / be_min)**(-n + 1.0) - (be_max / be_min)**(-n + 1.0)) / (n - 1.0) +
        kappa_high * be_min * c_cgs * rho0 *
        ((be_array / be_min)**(-n + 1.0) - (be_kappa / be_min)**(-n + 1.0)) / (n - 1.0))

    tau_high = (
        kappa_low * be_min * c_cgs * rho0 *
        ((be_array / be_min)**(-n + 1.0) - (be_max / be_min)**(-n + 1.0)) / (n - 1.0))

    taus = xp.where(mask_low, tau_low, tau_high)

    dMs = 4.0 * pi * vej**3.0 * rho0 * (be_array / be_min)**(-n + 2.0) * dbe / be_min
    tds = taus * be_array
    dt = t_array[1] - t_array[0]
    Eins0 = xp.zeros_like(be_array)

    sortidx = xp.argsort(taus)
    log_taus_sorted = xp.log(taus[sortidx])
    log_bes_sorted = xp.log(be_array[sortidx])

    def scan_fn(Eins, t):
        heat_th0 = _heating_rate(t)
        heat_th1 = _heating_rate(t + 0.5 * dt)
        heat_th2 = _heating_rate(t + dt)

        t_dif1 = tds / t
        heat1 = dMs * heat_th0
        L1, dE1 = compute_luminosity(Eins, t, t_dif1, be_array, heat1, dt)

        E2 = Eins + 0.5 * dE1
        t_dif2 = tds / (t + 0.5 * dt)
        heat2 = dMs * heat_th1
        L2, dE2 = compute_luminosity(E2, t + 0.5 * dt, t_dif2, be_array, heat2, dt)

        E3 = Eins + 0.5 * dE2
        t_dif3 = tds / (t + 0.5 * dt)
        heat3 = dMs * heat_th1
        L3, dE3 = compute_luminosity(E3, t + 0.5 * dt, t_dif3, be_array, heat3, dt)

        E4 = Eins + dE3
        t_dif4 = tds / (t + dt)
        heat4 = dMs * heat_th2
        L4, dE4 = compute_luminosity(E4, t + dt, t_dif4, be_array, heat4, dt)

        dE_total = (dE1 + 2*dE2 + 2*dE3 + dE4) / 6.0
        Eins_new = Eins + dE_total
        Ltot = xp.sum((L1 + 2*L2 + 2*L3 + L4) / 6.0)

        t += dt
        tt2 = t * t
        log_tt2 = xp.log(tt2)

        def branch_interp(_):
            Eins_sorted = xp.sort(Eins_new)
            log_E_sorted = xp.log(Eins_sorted )#+ 1e-20)
            log_be = jax_interp(log_tt2, log_taus_sorted, log_bes_sorted)
            r = xp.exp(log_be) * c_cgs * t
            Eint = xp.exp(jax_interp(log_tt2, log_taus_sorted, log_E_sorted))
            return r, Eint

        def branch_right(_):
            r = be_array[-1] * c_cgs * t
            return r, Eins_new[-1]

        def branch_left(_):
            r = be_array[0] * c_cgs * t
            return r, Eins_new[0]

        r, Eint = lax.cond(
            (taus[0]/tt2 > 1.0) & (taus[-1]/tt2 < 1.0),
            branch_interp,
            lambda _: lax.cond(
                taus[-1]/tt2 > 1.0,
                branch_right,
                branch_left,
                operand=None),
            operand=None)

        tmp = Ltot / (4. * pi * sb_cgs * r * r)
        temp = tmp**0.25

        return Eins_new, (t / u_cgs, Ltot, temp, r)

    _, (ts, Ls, temps, Rph) = lax.scan(scan_fn, Eins0, t_array)
    return ts, Ls, temps, Rph

def _calc_lightcurve_jax(Mej, vej,
                    alpha_max, alpha_min, n,
                    kappa_low, kappa_high,
                    be_kappa, be_array, Nbeta, be_min, be_max, dbe, t_array):

    mask = thinning_mask(t_array)
    ts, Ls, temps, Rph = _calc_lightcurve_jax_raw(Mej, vej,
                        alpha_max, alpha_min, n,
                        kappa_low, kappa_high,
                        be_kappa, be_array, Nbeta, be_min, be_max, dbe, t_array)

    ts, Ls, temps, Rph = ts[mask], Ls[mask], temps[mask], Rph[mask]
    return ts, Ls, temps, Rph




def _calc_lightcurve_khr(times, mass, velocities, opacities, n):

    pass
