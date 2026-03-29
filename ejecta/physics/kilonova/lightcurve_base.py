import pkg_resources
import extinction
import pyphot
from pyphot import unit, Filter
from gemma.backend import get_arraylib, get_mathlib, jit_if_available
from gemma.utils import bilby_conversion as bilby_conv
from gemma.utils.units_conversion import*
import numpy as np
from functools import partial
import scipy.interpolate
import time
# Setup backend aliases
xp = get_arraylib()
mp = get_mathlib()
jit = jit_if_available()


#@jit
def safe_log10(arr):
    arr = xp.where(arr <= 0, 1e-300, arr)
    return xp.log10(arr)


class EMbase:

    def __init__(self, hotokezaka_model=False):
        self.filter_library = pyphot.get_library()
        self.hotokezaka_model = hotokezaka_model
        self.params = None
        self._filter_cache = {}  # initialize empty cache
        #self.use_extinction = False  # default
        if self.sample_gw_parameters:
            if self.ejecta_function is not None:
                base_func = self.ejecta_function.func if hasattr(self.ejecta_function, "func") else self.ejecta_function
                if "bns" in base_func.__name__:
                    self._preprocess_gw = self._preprocess_bns_gw_parameters
                elif "nsbh" in base_func.__name__:
                    self._preprocess_gw = self._preprocess_nsbh_gw_parameters
                else:
                    raise ValueError("ejecta_function name must contain 'bns' or 'nsbh'")
            else:
                raise ValueError("You must provide an ejecta_function if sample_gw_parameters=True.")


            if self.gw_param_mode not in ["mass", "chirp_mass"]:
                raise ValueError("gw_param_mode must be either 'mass' or 'chirp_mass'")

        if getattr(self, "is_mosfit", False):
            ejecta_func = getattr(self, "ejecta_function", None)

            if ejecta_func is not None:
                if isinstance(ejecta_func, partial):
                    use_mosfit_model = ejecta_func.keywords.get("use_mosfit_model", False)
                else:
                    use_mosfit_model = getattr(ejecta_func, "use_mosfit_model", False)

                if not use_mosfit_model:
                    raise ValueError("You are using the MOSFiT model, but the ejecta_function does not have "
                        "`use_mosfit_model=True`. Please update the ejecta function.")




    def bounds_check(self, param_dict):
        for name in self.parameter_names:
            val = param_dict[name]
            low, high = self.parameter_bounds[name]
            if not (low <= val <= high):
                raise ValueError(f"[Bound check failed] '{name}' = {val:.4g} is outside [{low:.4g}, {high:.4g}]")


    def update_model(self, params, check_bounds=False):
        if check_bounds:
            self.bounds_check(params)
        if self.sample_gw_parameters:
            preprocessed = {**self._preprocess_gw(params), **self.ejecta_function.keywords}
            # print("YYYY", preprocessed)
            ejecta = self.ejecta_function(**preprocessed, preprocessed=True)
            #print("Yves", ejecta)
            params = {**params,
                      "m_ejecta_dyn": ejecta["m_ejecta_dyn"],
                      "v_ejecta_dyn": ejecta["v_ejecta_dyn"],
                      "m_disk": ejecta["m_ejecta_disk"],
                      "m_ejecta_wind": ejecta["m_ejecta_wind"],
                      "m_tov":ejecta['m_tov'],
                      "threshold_mass":ejecta['threshold_mass'],
                      "compactness_1": ejecta['compactness_1'],
                      "compactness_2": ejecta['compactness_2'],
                      "mass_1": preprocessed['mass_1'],
                      "mass_2": preprocessed['mass_2']}
        self.params = params
        self.generate_light(self.params)

    def get_magnitudes_pivot(self, times, filters):
        if self.params is None:
            raise TypeError("You haven't given any parameters to the kilonova model yet! "
                            "Use the `update_model` method first.")

        self.times = xp.asarray(times)
        self.filters = filters
        D_cm = self.params["luminosity_distance"] * Mpc_in_cm
        self.magnitudes = {}

        for f in self.filters:

            if not isinstance(self.times, xp.ndarray):
                times_for_filter = xp.asarray(self.times[f])
            else:
                times_for_filter = self.times

            ff = self.filter_library[f]
            lpivot_m = ff.lpivot.value * 1e-9  # meters
            nu_hz = c_cgs * 1e2 / lpivot_m  # Hz (c_cgs is in cm/s -> times 1e2 to m/s)

            corr = xp.pi * self.blackbody_lam_value(self.Temperature, lpivot_m) / (
                sigma_sb_cgs * (self.Temperature**4)
            )

            flux = (corr * self.BolometricLuminosity) / (4.0 * xp.pi * D_cm**2)  # erg/s/cm^2

            flux_per_AA = flux / (1e8)  # erg/s/cm^2/Å

            mag = -2.5 * xp.log10(flux_per_AA) - ff.AB_zero_mag

            if len(mag) != len(self.times):
                mag_times = self._evaluate_magnitude_times(mag, times_for_filter, selt._t)
            else:
                mag_times = mag

            self.magnitudes[f] = mag_times

        return self.magnitudes

    def get_magnitudes(self, times, filters):

        if self.params is None:
            raise TypeError("You haven't given any parameters to the kilonova model yet! "
                            "Use the `update_model` method first.")

        self.times = times
        self.filters = filters
        spectra = self.get_spectra()
        self.magnitudes = {}
        for f in self.filters:
            if not isinstance(self.times, xp.ndarray):
                times_for_filter = xp.asarray(self.times[f])
            else:
                times_for_filter = self.times
            ff = self.get_cached_filter(f)
            # Detach from JAX before calling non-JAX-safe methods
            model_wavelengths_np = np.asarray(self.model_wavelengths)  # Detach from JAX tracer
            spectra_np = np.asarray(xp.nan_to_num(spectra))            # Detach + sanitize
            flux = ff.get_flux(model_wavelengths_np * unit["AA"],
                                spectra_np * unit["flam"])
            flux = xp.array(flux)
            #flux = xp.clip(flux, 1e-300, xp.inf)
            #print("FLUX", flux)
            flux = flux.astype(xp.float64)
            zeropoint = xp.array(ff.AB_zero_mag, dtype=xp.float64)
            mag = -2.5 * safe_log10(flux) - zeropoint

            mag = xp.array(mag)
            #print("MAG", mag)

            if len(mag) != len(times_for_filter):
                #print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH",self._t)
                mag_times = self._evaluate_magnitude_times(mag, times_for_filter, self._t)
            else:
                mag_times = mag
            #print("Yves", mag_times)

            self.magnitudes[f] = mag_times

        return self.magnitudes
    def get_cached_filter(self, filter_name):
        """
        Load and cache a filter response curve, with optional extinction applied.

        The cache key includes filter name and extinction parameters if used.
        """
        if not hasattr(self, "_filter_cache"):
            self._filter_cache = {}

        if filter_name in self._filter_cache:
            return self._filter_cache[filter_name]
        #### Be careful, when sampling, the extention parameters won't change. This need to be modify to make that happen

        file_name = pkg_resources.resource_filename("gemma", f"filters/{filter_name}.dat")
        wavelength_data, flux_data = np.loadtxt(file_name).T  # always use NumPy here

        if self.use_extinction:
            magnitude_extinction = extinction.fitzpatrick99(wavelength_data, self.params["a_v"], self.params["r_v"])
            flux_data = extinction.apply(magnitude_extinction, flux_data)

        ff = Filter(
            wavelength_data * unit["AA"],
            flux_data,
            name=filter_name,
            dtype="photon",
            unit="Angstrom",
        )

        self._filter_cache[filter_name] = ff
        return ff



    def get_spectra(self):
        self.D = self.params["luminosity_distance"] * Mpc_in_cm
        spectra = self.generate_spectra()
        #print("SPECTRA", spectra)
        return spectra

    def _evaluate_magnitude_times(self, magnitudes, data_times, model_times):
        """
        Interpolates model magnitudes to the observation times.

        If the model_times array is empty (e.g., due to a failed or trivial lightcurve),
        this function returns an array of np.inf values to signal "no emission".

        IMPORTANT:
        Using np.inf will cause the log-likelihood to return -np.inf due to
        (inf - data)^2, which is the correct behavior in rejection sampling.

        You may optionally replace np.inf with a large finite value (e.g., 50.0)
        to keep downstream math stable in some samplers or frameworks.

        Parameters
        ----------
        magnitudes : array-like
            The model magnitudes evaluated at model_times.
        data_times : array-like
            The times at which magnitudes are required (observation times).
        model_times : array-like
            The times corresponding to the model magnitudes.

        Returns
        -------
        array
            Interpolated magnitudes at the data_times, or np.inf if interpolation fails.
        """
        if len(model_times) == 0:
            # Interpolation impossible — return "infinite" magnitudes (i.e., zero flux)
            return xp.ones_like(data_times) * xp.inf

        func = scipy.interpolate.interp1d(model_times, magnitudes, bounds_error=False, fill_value=50)
        return func(data_times)

    @staticmethod
    @jit
    def blackbody_nu_value(temperature, nu_hz):
        """
        Blackbody spectrum per unit frequency [erg/s/cm^2/Hz/sr].
        Inputs: temperature [K], nu [Hz].
        """
        return (2.0 * h_cgs * nu_hz**3) / (c_cgs**2) / (xp.exp(h_cgs * nu_hz / (k_B_cgs * temperature)) - 1.0)

    @staticmethod
    @jit
    def blackbody_lam_value(temperature, lam_m):
        """
        Blackbody spectrum per unit wavelength [erg/s/cm^2/cm/sr].
        Inputs: temperature [K], lambda [m].
        """
        lam_cm = lam_m * 100.0  # meters to centimeters
        return (2.0 * h_cgs * c_cgs**2 / lam_cm**5) / (xp.exp(h_cgs * c_cgs / (lam_cm * k_B_cgs * temperature)) - 1.0)

    def blackbody_nu(self, temperature, nu_hz):
        return self.blackbody_nu_value(temperature, nu_hz)

    def blackbody_lam(self, temperature, lam_m):
        return self.blackbody_lam_value(temperature, lam_m)


    def _preprocess_bns_gw_parameters(self, params):
        if self.gw_param_mode == "chirp_mass":

            m1, m2 = bilby_conv.chirp_mass_and_mass_ratio_to_component_masses(
                params["chirp_mass"], params["mass_ratio"]
            )
            chirp_mass = params["chirp_mass"]
            mass_ratio = params["mass_ratio"]
        elif self.gw_param_mode == "mass":
            m1, m2 = params["mass_1"], params["mass_2"]
            chirp_mass = bilby_conv.component_masses_to_chirp_mass(m1, m2)
            mass_ratio = m2 / m1
        else:
            raise ValueError(f"Unknown gw_param_mode: {self.gw_param_mode}")

        # Tidal deformabilities
        if "lambda_tilde" in params and ("lambda_1" not in params or "lambda_2" not in params):
            lambda_1, lambda_2 = bilby_conv.lambda_tilde_to_lambda_1_lambda_2(
                params["lambda_tilde"], m1, m2
            )
            lambda_tilde = params["lambda_tilde"]
        else:
            lambda_1, lambda_2 = params["lambda_1"], params["lambda_2"]
            lambda_tilde = bilby_conv.lambda_1_lambda_2_to_lambda_tilde(
                lambda_1, lambda_2, m1, m2
            )
        return {
            "mass_1": m1,
            "mass_2": m2,
            "chirp_mass": chirp_mass,
            "mass_ratio": mass_ratio,
            "lambda_1": lambda_1,
            "lambda_2": lambda_2,
            "lambda_tilde": lambda_tilde,
        }



    def _preprocess_nsbh_gw_parameters(self, params):

        if self.gw_param_mode == "chirp_mass":
            m1, m2 = bilby_conv.chirp_mass_and_mass_ratio_to_component_masses(
                params["chirp_mass"], params["mass_ratio"]
            )
            chirp_mass = params["chirp_mass"]
            mass_ratio = params["mass_ratio"]
        elif self.gw_param_mode == "mass":
            m1, m2 = params["mass_1"], params["mass_2"]
            chirp_mass = bilby_conv.component_masses_to_chirp_mass(m1, m2)
            mass_ratio = m2 / m1
        else:
            raise ValueError(f"Unknown gw_param_mode: {self.gw_param_mode}")

        return {
            "mass_1": m1,
            "mass_2": m2,
            "chirp_mass": chirp_mass,
            "mass_ratio": mass_ratio,
            "lambda_2": params["lambda_2"],
            "chi_1": params["chi_1"],
        }
