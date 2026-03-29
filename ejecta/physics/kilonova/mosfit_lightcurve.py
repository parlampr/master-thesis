# mosfit_lightcurve.py
from gemma.backend import get_arraylib, jit_if_available, get_backend_name, get_erfc
from gemma.utils.units_conversion import*
from .lightcurve_base import EMbase
import astropy.units as u
from scipy.interpolate import interp1d

# Backend setup
xp = get_arraylib()
jit = jit_if_available()
backend_name = get_backend_name()
if backend_name == "jax":
    import jax
    from jax import lax
    def mosfit_bns_quantities(*args, **kwargs):
        keys = [
        "m_ejecta_red", "m_ejecta_blue", "m_ejecta_purple",
        "v_ejecta_red", "v_ejecta_blue", "v_ejecta_purple",
        "v_ejecta_mean", "kappa_purple", "kappa_mean"]
        values = mosfit_bns_quantities_jax(*args,**kwargs)
        #print("JAX", dict(zip(keys, values)))
        return dict(zip(keys, values))


    def mosfit_diffusion(*args, **kwargs):
        return _mosfit_diffusion_jax(*args, **kwargs)
else:
    def mosfit_bns_quantities(*args, **kwargs):
        keys = [
        "m_ejecta_red", "m_ejecta_blue", "m_ejecta_purple",
        "v_ejecta_red", "v_ejecta_blue", "v_ejecta_purple",
        "v_ejecta_mean", "kappa_purple", "kappa_mean"]
        values = mosfit_bns_quantities_numba(*args,**kwargs)
        #print("NUMBA", dict(zip(keys, values)))
        return dict(zip(keys, values))
        #return mosfit_bns_quantities_numba(*args, **kwargs)

    def mosfit_diffusion(*args, **kwargs):
        return _mosfit_diffusion_numba(*args, **kwargs)




NSBH_COMPONENTS = ['dynamic', 'wind', 'magnetic']
BNS_COMPONENTS = ['cocoon', 'dynamic', 'wind', 'magnetic']
from gemma.backend import get_arraylib, jit_if_available

xp = get_arraylib()

# Grids and table values
barnes_v = xp.array([0.1, 0.2, 0.3, 0.4])
barnes_M = xp.array([1.0e-3, 5.0e-3, 1.0e-2, 5.0e-2, 1.0e-1])

barnes_a = xp.array([
    [2.01, 4.52, 8.16, 16.3],
    [0.81, 1.9, 3.2, 5.0],
    [0.56, 1.31, 2.19, 3.0],
    [0.27, 0.55, 0.95, 2.0],
    [0.20, 0.39, 0.65, 0.9],
])

barnes_b = xp.array([
    [0.28, 0.62, 1.19, 2.4],
    [0.19, 0.28, 0.45, 0.65],
    [0.17, 0.21, 0.31, 0.45],
    [0.10, 0.13, 0.15, 0.17],
    [0.06, 0.11, 0.12, 0.12],
])

barnes_d = xp.array([
    [1.12, 1.39, 1.52, 1.65],
    [0.86, 1.21, 1.39, 1.5],
    [0.74, 1.13, 1.32, 1.4],
    [0.6, 0.9, 1.13, 1.25],
    [0.63, 0.79, 1.04, 1.5],])

@jit
def clip_scalar(x, x_min, x_max):
    return xp.minimum(xp.maximum(x, x_min), x_max)


@jit
def bilinear_interp_2d(M, v, M_grid, v_grid, values):
    i = xp.searchsorted(M_grid, M, side="right") - 1
    j = xp.searchsorted(v_grid, v, side="right") - 1

    i = clip_scalar(i, 0, len(M_grid) - 2)
    j = clip_scalar(j, 0, len(v_grid) - 2)

    M0, M1 = M_grid[i], M_grid[i + 1]
    v0, v1 = v_grid[j], v_grid[j + 1]

    q11 = values[i, j]
    q21 = values[i + 1, j]
    q12 = values[i, j + 1]
    q22 = values[i + 1, j + 1]

    denom = (M1 - M0) * (v1 - v0)
    denom = xp.where(denom == 0, 1.0, denom)

    interp_val = (
        q11 * (M1 - M) * (v1 - v) +
        q21 * (M - M0) * (v1 - v) +
        q12 * (M1 - M) * (v - v0) +
        q22 * (M - M0) * (v - v0)
    ) / denom

    return interp_val

# Wrap functions for a, b, d interpolation
@jit
def therm_func_a(M, v):
    return bilinear_interp_2d(M, v, barnes_M, barnes_v, barnes_a)

@jit
def therm_func_b(M, v):
    return bilinear_interp_2d(M, v, barnes_M, barnes_v, barnes_b)

@jit
def therm_func_d(M, v):
    return bilinear_interp_2d(M, v, barnes_M, barnes_v, barnes_d)


class MOSFiT(EMbase):

    def __init__(self, components=['wind', 'dynamic'], sample_gw_parameters=False,
                 gw_param_mode="chirp_mass", ejecta_function=None,
                 model_name=None, use_extinction=False):
        """
        Initialize the kilonova model class with optional ejecta functions and GW parameter handling.

        Parameters
        ----------
        components : list of str
            Ejecta components to include in the model (e.g., ['wind', 'dynamic']).
        sample_gw_parameters : bool
            If True, allows sampling of GW parameters like chirp mass or mass ratio.
        gw_param_mode : str
            Mode for GW parameterization. Typically 'chirp_mass' or 'm1_m2'.
        ejecta_function : callable or None
            Function that estimates ejecta properties from progenitor parameters.
            Must match the `model_name` ('bns' or 'nsbh') or will raise an error.
        model_name : str
            Model identifier; should be either 'bns' or 'nsbh'.
        use_extinction : bool
            If True, includes extinction effects in synthetic light curves.
        """

        self.is_mosfit = True

        # Define model wavelengths in Angstroms and convert to CGS (cm)
        self.model_wavelengths = xp.linspace(100.0, 99900.0, 500)
        self.model_wavelengths_cgs = (self.model_wavelengths * u.AA).to(u.cm).value

        # For JAX backend, ensure input arrays are compatible
        if backend_name == "jax":
            self.model_wavelengths_cgs = xp.array(self.model_wavelengths_cgs)

        # Set core attributes
        self.components = components
        self.use_extinction = use_extinction
        self.sample_gw_parameters = sample_gw_parameters
        self.ejecta_function = ejecta_function
        self.gw_param_mode = gw_param_mode
        self.model_name = model_name
        self.number_components = len(self.components)  # Will be updated based on model type and components

        # Assign light curve generation function based on model name
        if model_name == "bns":
            self.generate_light = self.generate_light_bns
        elif model_name == "nsbh":
            self.generate_light = self.generate_light_nsbh
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")


        # If no ejecta function is provided, assign default light generation function
        if self.ejecta_function is None:
            self.number_components += 1  # Default model counts as one component

        else:
            # Extract the base function, even if wrapped (e.g., via partial)
            base_func = (self.ejecta_function.func
                         if hasattr(self.ejecta_function, "func")
                         else self.ejecta_function)
            func_name = base_func.__name__.lower()

            # Ensure that the ejecta function matches the declared model type
            if self.model_name not in func_name:
                raise ValueError(f"Ejecta function '{func_name}' does not match model_name '{self.model_name}'.")

            # Increment number of components based on both the component type and model type
            if 'dynamic' in self.components and "bns" in func_name:
                self.number_components += 1
            if 'wind' in self.components and "nsbh" in func_name:
                self.number_components += 1

        self._times = xp.logspace(-3, 2, 100)
        self._t = self._times

        self.parameter_names = []
        self.bounds = {}

        if self.sample_gw_parameters:
            self.parameter_names += [
                "mass_1", "mass_2",
                "compactness_1", "compactness_2",
                "m_tov", "threshold_mass",
                "alpha",
                "kappa_red", "kappa_blue",
                "m_dyn", "m_ejecta_wind",
                "cos_theta_open", "cos_theta_jn",
                "temperature", "kappa_gamma",
            ]
        else:
            self.parameter_names += [
                "m_ejecta_dyn", "m_ejecta_wind", "fraction_red",
                "v_ejecta_red", "v_ejecta_blue", "v_ejecta_purple", "v_ejecta_mean",
                "kappa_red", "kappa_blue", "kappa_purple", "kappa_mean",
                "cos_theta_open", "cos_theta_jn",
                "temperature", "kappa_gamma",
            ]

        # Add component-specific parameters
        if "cocoon" in self.components:
            self.parameter_names += ["cos_theta_cocoon", "shocked_fraction", "pl_s", "t_shock"]

        if "magnetic" in self.components:
            self.parameter_names += ["period_spin", "magnetic_field", "albedo", "pair_multiplicity", "epsilon_therm", "m_disk", "R16"]

        if self.model_name == "nsbh":
            self.parameter_names += ["cos_theta_open_dyn"]

        #print("Yves0", sample_gw_parameters,self.sample_gw_parameters )
        #print("Yves0", )

        super(MOSFiT, self).__init__()


    def generate_light_bns(self, params):
        """
        Generate light curves and photosphere properties for BNS merger components
        using a backend-compatible implementation (NumPy, JAX, Numba).

        Stores:
        - self.RadiusPhotosphere: list of arrays
        - self.TemperaturePhotosphere: list of arrays
        - self.params: updated parameter dictionary (with derived quantities)

        Assumes:
        - self._times, self.backend, self.xp, self.components, self.em_ejecta are set.
        - self.aspherical_kilonova returns projected and reference areas.
        """

        self.params = params

        times = self._times
        #print("Yves", self.sample_gw_parameters)
        if self.sample_gw_parameters:
            self.params.update(mosfit_bns_quantities(params["mass_1"],
                                                params["mass_2"],
                                                params["cos_theta_open"],
                                                params["m_ejecta_dyn"],
                                                params["m_ejecta_wind"],
                                                params["compactness_1"],
                                                params["compactness_2"],
                                                params["m_tov"],
                                                params["threshold_mass"],
                                                params["alpha"],
                                                params["kappa_red"],
                                                params["kappa_blue"]))
        else:
            self.params["m_ejecta_red"] = params["m_ejecta_dyn"] * params["fraction_red"]
            self.params["m_ejecta_blue"] = params["m_ejecta_dyn"] * (1.0 - params["fraction_red"])
            self.params["m_ejecta_purple"] = params["m_ejecta_wind"]

        # Get projected and reference areas
        self.area_blue, self.area_blue_ref, self.area_red, self.area_red_ref = aspherical_kilonova(
            self.params["cos_theta_open"], self.params["cos_theta_jn"])

        m_ejectas, v_ejectas, kappas = [], [], []
        areas_projected, areas_reference = [], []

        self.RadiusPhotosphere = []
        self.TemperaturePhotosphere = []

        if "cocoon" in self.components:
            cocoon_lums = mosfit_shock_cocoon(
                self.params["m_ejecta_blue"], self.params["v_ejecta_blue"], self.params["kappa_blue"],
                self.params["cos_theta_cocoon"], self.params["shocked_fraction"], self.params["pl_s"],
                self.params["t_shock"], times)
            r_ph, t_ph = mosfit_shock_photosphere(
                self.params["m_ejecta_blue"], self.params["v_ejecta_blue"], self.params["kappa_blue"],
                self.params["shocked_fraction"], self.params["pl_s"], times, cocoon_lums)
            self.RadiusPhotosphere.append(r_ph)
            self.TemperaturePhotosphere.append(t_ph)

        if "dynamic" in self.components:
            m_ejectas.extend([self.params["m_ejecta_blue"], self.params["m_ejecta_red"]])
            v_ejectas.extend([self.params["v_ejecta_blue"] * c_km_s, self.params["v_ejecta_red"] *c_km_s])
            kappas.extend([self.params["kappa_blue"], self.params["kappa_red"]])
            areas_projected.extend([self.area_blue, self.area_red])
            areas_reference.extend([self.area_blue_ref, self.area_red_ref])

        if "wind" in self.components:
            m_ejectas.append(self.params["m_ejecta_purple"])
            v_ejectas.append(self.params["v_ejecta_purple"] * c_km_s)
            kappas.append(self.params["kappa_purple"])
            areas_projected.append(self.area_blue)
            areas_reference.append(self.area_blue_ref)

        #print("JECT", m_ejectas)

        for m_ej, v_ej, kappa, area_proj, area_ref in zip(
            m_ejectas, v_ejectas, kappas, areas_projected, areas_reference):
            #print("BERRRRRRRRRRRRRRRRRRRRRR")
            rproc_lums = mosfit_rprocess(m_ej, v_ej, times)
            #print("HERE", m_ej, v_ej)
            lums = mosfit_diffusion(m_ej, v_ej, kappa, self.params["kappa_gamma"],
                times, times, rproc_lums,
                area_proj, area_ref,
                aspherical_diffusion=True)
            #print("THERE", lums)

            r_ph, t_ph = mosfit_temperature(v_ej, self.params["temperature"], times, lums)
            self.RadiusPhotosphere.append(r_ph)
            self.TemperaturePhotosphere.append(t_ph)

        if "magnetic" in self.components:
            lums_magnetic = mosfit_bns_magnetar(self.params, self.em_ejecta, times)
            total_mass = self.em_ejecta.m_dyn + self.em_ejecta.m_ejecta_wind
            lums = mosfit_diffusion(
                total_mass, self.params["v_ejecta_mean"], self.params["kappa_mean"],
                self.params["kappa_gamma_magnetic"], times, times,
                lums_magnetic)
            r_ph, t_ph = mosfit_temperature(self.params["v_ejecta_mean"], self.params["temperature"], times, lums)
            self.RadiusPhotosphere.append(r_ph)
            self.TemperaturePhotosphere.append(t_ph)
        #print("Yves", self.params)


    def generate_light_nsbh(self, params):
        """
        Generates light curves for a neutron star–black hole (NSBH) merger using the Gompertz et al. 2023 model.
        Computes temperature and radius of the photosphere for each component in `self.components`.

        Parameters
        ----------
        params : dict
            Dictionary of physical parameters (ejecta masses, velocities, opacities, angles, etc.)
        """

        # Compute aspherical geometries
        area_blue, area_blue_ref, area_red, area_red_ref = aspherical_kilonova(
            params["cos_theta_open"], params["cos_theta_jn"])
        area_dyn, area_dyn_ref, area_thermal, area_thermal_ref, area_magnetic, area_magnetic_ref = aspherical_wind(
            params["cos_theta_open_dyn"], params["cos_theta_jn"],
            area_red, area_red_ref, area_blue, area_blue_ref)

        # Initialize lists for components
        m_ejectas, v_ejectas, kappas = [], [], []
        areas_projected, areas_reference = [], []

        # Dynamical ejecta
        if "dynamic" in self.components:
            m_ejectas.append(params["m_ejecta_dyn"])
            v_ejectas.append(params["v_ejecta_dyn"] * c_km_s)
            kappas.append(params["kappa_dyn"])
            areas_projected.append(area_dyn)
            areas_reference.append(area_dyn_ref)

        # Wind ejecta (blue + purple)
        if "wind" in self.components:
            log_mdisk = xp.log10(params["m_disk"])
            frac_blue = 0.20199 * log_mdisk + 1.12629
            m_blue = params["m_ejecta_wind"] * frac_blue
            m_purple = params["m_ejecta_wind"] * (1.0 - frac_blue)
            v_wind = 0.034 * c_km_s

            m_ejectas.extend([m_blue, m_purple])
            v_ejectas.extend([v_wind, v_wind])
            kappas.extend([params["kappa_thermal_blue"], params["kappa_thermal_purple"]])
            areas_projected.extend([area_thermal, area_thermal])
            areas_reference.extend([area_thermal_ref, area_thermal_ref])

        # Magnetic component
        if "magnetic" in self.components:
            m_mag = params["fraction_magnetic"] * params["m_ejecta_wind"]
            v_mag = max(0.034 * c_km_s, 0.22 * params["fraction_magnetic"] * c_km_s)

            m_ejectas.append(m_mag)
            v_ejectas.append(v_mag)
            kappas.append(params["kappa_dyn"])
            areas_projected.append(area_magnetic)
            areas_reference.append(area_magnetic_ref)

        # Prepare output
        radius_list, temperature_list = [], []

        # Loop over all ejecta components
        for i, (m, v, kappa, area_proj, area_ref) in enumerate(zip(m_ejectas, v_ejectas, kappas, areas_projected, areas_reference)):
            L_rproc = mosfit_rprocess(m, v, self._times)

            # Base luminosity from diffusion
            L_diff = mosfit_diffusion(
                m, v, kappa, params["kappa_gamma"],
                self._times, self._times, L_rproc,
                area_proj, area_ref, aspherical_diffusion=True
            )

            # Interaction component (see paper logic)
            if (
                ("wind" in self.components and "dynamic" in self.components and i == 2)
                or ("wind" in self.components and "magnetic" in self.components and i == 1)
                or ("wind" in self.components and self.number_components == 2 and i == 1)
            ):
                L_diff += mosfit_diffusion(
                    m_blue, v, params["kappa_thermal_blue"], params["kappa_gamma"],
                    self._times, self._times, L_rproc,
                    area_proj, area_ref, aspherical_diffusion=True
                )

            # Compute photosphere radius and temperature
            R_phot, T_phot = mosfit_temperature(v, params["temperature"], self._times, L_diff)

            radius_list.append(R_phot)
            temperature_list.append(T_phot)

        # Store results
        self.RadiusPhotosphere = radius_list
        self.TemperaturePhotosphere = temperature_list



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

@jit
def blackbody_lam(temperature, lam):
    exp_arg = h_cgs * c_cgs / (lam * kB_cgs * temperature)
    return (2.0 * h_cgs * c_cgs**2 / (lam**5 * (xp.exp(exp_arg) - 1.0))) * 1e-8 # u.erg/u.s/u.cm**2/u.AA



@jit
def aspherical_kilonova(cos_theta_open, cos_theta_jn):
    """
    Compute the projected and reference areas of the blue and red regions of an aspherical kilonova
    as described in Darbha & Kasen (2020). This version is JAX- and Numba-compatible and purely functional.

    Parameters
    ----------
    cos_theta_open : float
        Cosine of the polar opening angle of the blue component.
    cos_theta_jn : float
        Cosine of the observer inclination angle.

    Returns
    -------
    area_blue : float
        Projected area of the blue component in the observer direction.
    area_blue_ref : float
        Reference projected area for face-on viewing angle.
    area_red : float
        Projected area of the red component in the observer direction.
    area_red_ref : float
        Reference projected area of the red component for face-on viewing angle.
    """
    # ct defines the sine of the opening angle (blue region extent)
    ct = xp.sqrt(1.0 - cos_theta_open**2)

    # Precompute reusable values
    one_minus_cos_jn_sq = 1.0 - cos_theta_jn**2
    sqrt_jn = xp.sqrt(one_minus_cos_jn_sq)
    sqrt_half = xp.sqrt(1.0 - 0.25)  # (1 - 0.5**2)

    # --- Projected area (top hemisphere) ---
    theta_p = xp.arccos(cos_theta_open / sqrt_jn)
    theta_d = xp.arctan(xp.sin(theta_p) / cos_theta_open * sqrt_jn / xp.abs(cos_theta_jn))
    top_val = (theta_p - xp.sin(theta_p) * xp.cos(theta_p)) - (ct * cos_theta_jn * (theta_d - xp.sin(theta_d) * xp.cos(theta_d) - pi))

    area_projected_top = xp.where(cos_theta_jn > ct, pi * ct * cos_theta_jn, top_val)

    # --- Projected area (bottom hemisphere) ---
    theta_p2 = xp.arccos(cos_theta_open / sqrt_jn)
    theta_d2 = xp.arctan(xp.sin(theta_p2) / cos_theta_open * sqrt_jn / xp.abs(cos_theta_jn))
    bot_val = (theta_p2 - xp.sin(theta_p2) * xp.cos(theta_p2)) + (
        ct * -cos_theta_jn * (theta_d2 - xp.sin(theta_d2) * xp.cos(theta_d2)))

    area_projected_bot = xp.where(-cos_theta_jn < -ct, 0.0, xp.maximum(bot_val, 0.0))

    area_projected = area_projected_top + area_projected_bot

    # --- Reference area (top hemisphere) for face-on view ---
    theta_p_ref = xp.arccos(cos_theta_open / sqrt_half)
    theta_d_ref = xp.arctan(xp.sin(theta_p_ref) / cos_theta_open * sqrt_half / 0.5)
    ref_top_val = (theta_p_ref - xp.sin(theta_p_ref) * xp.cos(theta_p_ref)) - (
        ct * 0.5 * (theta_d_ref - xp.sin(theta_d_ref) * xp.cos(theta_d_ref) - pi))

    area_reference_top = xp.where(0.5 > ct, pi * ct * 0.5, ref_top_val)

    # --- Reference area (bottom hemisphere) ---
    theta_p2_ref = xp.arccos(cos_theta_open / sqrt_half)
    theta_d2_ref = xp.arctan(xp.sin(theta_p2_ref) / cos_theta_open * sqrt_half / 0.5)
    ref_bot_val = (theta_p2_ref - xp.sin(theta_p2_ref) * xp.cos(theta_p2_ref)) + (
        ct * -0.5 * (theta_d2_ref - xp.sin(theta_d2_ref) * xp.cos(theta_d2_ref)))

    area_reference_bot = xp.where(-0.5 < -ct, 0.0, ref_bot_val)

    area_reference = area_reference_top + area_reference_bot

    # --- Final outputs ---
    area_blue = area_projected
    area_blue_ref = area_reference
    area_red = pi - area_blue
    area_red_ref = pi - area_blue_ref

    return area_blue, area_blue_ref, area_red, area_red_ref


@jit
def aspherical_wind(cos_theta_open_dyn, cos_theta_jn, area_red, area_red_ref, area_blue, area_blue_ref):
    """
    Compute projected and reference areas for an aspherical wind ejecta geometry
    following Darbha & Kasen (2020), compatible with JAX and Numba.

    Parameters
    ----------
    cos_theta_open_dyn : float
        Cosine of the dynamical opening angle for the wind ejecta.
    cos_theta_jn : float
        Cosine of the observer inclination angle.
    area_red : float
        Red component area (projected).
    area_red_ref : float
        Red component reference area (face-on).
    area_blue : float
        Blue component area (projected).
    area_blue_ref : float
        Blue component reference area (face-on).

    Returns
    -------
    area_dyn : float
        Projected area of the dynamical (wind) ejecta component.
    area_dyn_ref : float
        Reference area of the dynamical component.
    area_thermal : float
        Projected area of the thermal ejecta (red - dyn).
    area_thermal_ref : float
        Reference thermal area.
    area_magnetic : float
        Magnetic component area (same as blue).
    area_magnetic_ref : float
        Reference magnetic area (same as blue).
    """
    ct = xp.sqrt(1.0 - cos_theta_open_dyn**2)
    one_minus_cos_jn_sq = 1.0 - cos_theta_jn**2
    sqrt_jn = xp.sqrt(one_minus_cos_jn_sq)
    sqrt_half = xp.sqrt(1.0 - 0.25)

    # Top projected
    theta_p = xp.arccos(cos_theta_open_dyn / sqrt_jn)
    theta_d = xp.arctan(xp.sin(theta_p) / cos_theta_open_dyn * sqrt_jn / xp.abs(cos_theta_jn))
    top_val = (theta_p - xp.sin(theta_p) * xp.cos(theta_p)) - (
        ct * cos_theta_jn * (theta_d - xp.sin(theta_d) * xp.cos(theta_d) - pi))


    area_projected_top = xp.where(cos_theta_jn > ct, pi * ct * cos_theta_jn, top_val)

    # Bottom projected
    theta_p2 = xp.arccos(cos_theta_open_dyn / sqrt_jn)
    theta_d2 = xp.arctan(xp.sin(theta_p2) / cos_theta_open_dyn * sqrt_jn / xp.abs(cos_theta_jn))
    bot_val = (theta_p2 - xp.sin(theta_p2) * xp.cos(theta_p2)) + (
        ct * -cos_theta_jn * (theta_d2 - xp.sin(theta_d2) * xp.cos(theta_d2)))


    area_projected_bot = xp.where(-cos_theta_jn < -ct, 0.0, xp.maximum(bot_val, 0.0))

    area_projected = area_projected_top + area_projected_bot

    # Reference top
    theta_p_ref = xp.arccos(cos_theta_open_dyn / sqrt_half)
    theta_d_ref = xp.arctan(xp.sin(theta_p_ref) / cos_theta_open_dyn * sqrt_half / 0.5)
    ref_top_val = (theta_p_ref - xp.sin(theta_p_ref) * xp.cos(theta_p_ref)) - (
        ct * 0.5 * (theta_d_ref - xp.sin(theta_d_ref) * xp.cos(theta_d_ref) - pi))


    area_reference_top = xp.where(0.5 > ct, pi * ct * 0.5, ref_top_val)

    # Reference bottom
    theta_p2_ref = xp.arccos(cos_theta_open_dyn / sqrt_half)
    theta_d2_ref = xp.arctan(xp.sin(theta_p2_ref) / cos_theta_open_dyn * sqrt_half / 0.5)
    ref_bot_val = (theta_p2_ref - xp.sin(theta_p2_ref) * xp.cos(theta_p2_ref)) + (
        ct * -0.5 * (theta_d2_ref - xp.sin(theta_d2_ref) * xp.cos(theta_d2_ref)))

    area_reference_bot = xp.where(-0.5 < -ct, 0.0, ref_bot_val)

    area_reference = area_reference_top + area_reference_bot

    # Final computed areas
    area_dyn = pi - area_projected
    area_dyn_ref = pi - area_reference
    area_thermal = area_red - area_dyn
    area_thermal_ref = area_red_ref - area_dyn_ref
    area_magnetic = area_blue
    area_magnetic_ref = area_blue_ref

    return area_dyn, area_dyn_ref, area_thermal, area_thermal_ref, area_magnetic, area_magnetic_ref

@jit
def trapz_axis1(y, x):
    """
    Manual trapezoidal integration along axis=1.
    Assumes y and x are (n_samples, n_points)
    """
    n_samples, n_points = y.shape
    result = xp.zeros(n_samples)
    for i in range(n_samples):
        for j in range(n_points - 1):
            dx = x[i, j+1] - x[i, j]
            avg = 0.5 * (y[i, j] + y[i, j+1])
            result[i] += dx * avg
    return result

# More reliable check
is_jax_backend = xp.__name__ == "jax.numpy"
def flexible_jit(static_argnames=()):
    def decorator(fn):
        if is_jax_backend:
            return jit(fn, static_argnames=static_argnames)
        else:
            return jit(fn)
    return decorator

@flexible_jit(static_argnames=["aspherical_diffusion"])
def _mosfit_diffusion_jax(
    m_ejecta,
    v_ejecta,
    kappa,
    kappa_gamma,
    time,
    dense_times,
    luminosity,
    area_projected=None,
    area_reference=None,
    aspherical_diffusion=False):

    diffusion_constant = 2.0 * M_sun_cgs / (13.7 * c_cgs * km_cgs)
    trapping_constant = 3.0 * M_sun_cgs / (4.0 * pi * km_cgs**2)

    tau_diff = xp.sqrt(diffusion_constant * kappa * m_ejecta / v_ejecta) / day_cgs + 1e-30  #  To avoid 0.0 division error
    #print("JJJJJJJ JAX", tau_diff, diffusion_constant, kappa, m_ejecta,  v_ejecta, day_cgs)
    tau_trap = (trapping_constant * kappa_gamma * m_ejecta / v_ejecta**2) / day_cgs**2

    tb = dense_times[0]
    tmax = dense_times[-1]

    steps = 100
    lsp = xp.logspace(xp.log10(tau_diff / tmax + 1e-12) - 3.0, 0.0, steps // 2)
    xm = xp.sort(xp.concatenate((lsp, 1.0 - lsp)))
    xm = xp.clip(xm, 0.0, 1.0)

    def single_lum(t_i, correction_factor):
        dt = t_i - tb
        int_times = xp.clip(tb + dt * xm, tb, tmax)
        te2 = int_times[-1] ** 2
        int_lums = linear_interp_1d(int_times, dense_times, luminosity)
        integrand = int_lums * int_times * xp.exp((int_times**2 - te2) / tau_diff**2)
        integrand = xp.nan_to_num(integrand)
        dx = xp.diff(int_times)
        avg = 0.5 * (integrand[:-1] + integrand[1:])
        total = xp.sum(dx * avg)
        lum = -2.0 * xp.expm1(-tau_trap / te2) * total / tau_diff**2
        lum *= correction_factor
        return xp.where((t_i >= tb) & (t_i <= tmax), lum, 0.0)

    if aspherical_diffusion and (area_projected is not None and area_reference is not None):
        def compute_correction(t_i):
            t_norm = t_i / tau_diff / 0.59
            factor = (2.0 + t_norm) / (1.0 + xp.exp(t_norm))
            return 1.0 + 1.4 * factor * (area_projected / area_reference - 1.0)
        correction = jax.vmap(compute_correction)(time)
    else:
        correction = xp.ones_like(time)

    return jax.vmap(single_lum)(time, correction)
@jit
def _mosfit_diffusion_numba(
    m_ejecta,
    v_ejecta,
    kappa,
    kappa_gamma,
    time,
    dense_times,
    luminosity,
    area_projected,
    area_reference,
    aspherical_diffusion=False):
    diffusion_constant = 2.0 * M_sun_cgs / (13.7 * c_cgs * km_cgs)
    trapping_constant = 3.0 * M_sun_cgs / (4.0 * pi * km_cgs**2)

    tau_diff = (diffusion_constant * kappa * m_ejecta / v_ejecta)**0.5 / day_cgs + 1e-30  #  # To avoid 0.0 division error
    #print("JJJJJJJ numba", tau_diff, diffusion_constant, kappa, m_ejecta,  v_ejecta, day_cgs)
    tau_trap = (trapping_constant * kappa_gamma * m_ejecta / v_ejecta**2) / day_cgs**2

    tb = dense_times[0]
    tmax = dense_times[-1]

    steps = 100
    xm = xp.empty(steps)
    log_min = xp.log10(tau_diff / tmax + 1e-12) - 3.0
    for i in range(steps // 2):
        l = 10 ** (log_min + i * (0.0 - log_min) / (steps // 2 - 1))
        xm[i] = l
        xm[steps - i - 1] = 1.0 - l
    xm.sort()

    output = xp.zeros_like(time)
    for i in range(time.shape[0]):
        t_i = time[i]
        if tb <= t_i <= tmax:
            dt = t_i - tb
            int_times = xp.clip(tb + dt * xm, tb, tmax)
            te2 = int_times[-1] ** 2
            int_lums = linear_interp_1d(int_times, dense_times, luminosity)
            integrand = int_lums * int_times * xp.exp((int_times**2 - te2) / tau_diff**2)
            for j in range(integrand.shape[0]):
                if xp.isnan(integrand[j]):
                    integrand[j] = 0.0
            total = 0.0
            for j in range(len(int_times) - 1):
                dx = int_times[j + 1] - int_times[j]
                avg = 0.5 * (integrand[j] + integrand[j + 1])
                total += dx * avg
            lum = -2.0 * (xp.expm1(-tau_trap / te2)) * total / tau_diff**2

            if aspherical_diffusion and (area_projected is not None and area_reference is not None):
                t_norm = t_i / tau_diff / 0.59
                factor = (2.0 + t_norm) / (1.0 + xp.exp(t_norm))
                correction = 1.0 + 1.4 * factor * (area_projected / area_reference - 1.0)
                lum *= correction
            output[i] = lum
        else:
            output[i] = 0.0

    return output


@jit
def linear_interp_1d(x_query, x_grid, y_grid):
    """
    JIT-safe linear interpolation.

    Parameters
    ----------
    x_query : array_like
        Points to evaluate.
    x_grid : array_like
        Known x-values (must be increasing).
    y_grid : array_like
        Known y-values (same shape as x_grid).

    Returns
    -------
    y_query : array_like
        Interpolated values at x_query.
    """
    # Ensure monotonic grid
    n = x_grid.shape[0]
    idx = xp.clip(xp.searchsorted(x_grid, x_query) - 1, 0, n - 2)

    x0 = x_grid[idx]
    x1 = x_grid[idx + 1]
    y0 = y_grid[idx]
    y1 = y_grid[idx + 1]

    slope = (y1 - y0) / (x1 - x0)
    return y0 + slope * (x_query - x0)


@jit
def mosfit_temperature(v_ejecta, temperature_floor, times, luminosities):
    """
    Compute the photospheric radius and temperature using the MOSFiT-style prescription.

    Parameters
    ----------
    v_ejecta : float
        Ejecta velocity in km/s.
    temperature_floor : float
        Minimum (floor) temperature in Kelvin.
    times : array_like
        Observation times in days.
    luminosities : array_like
        Bolometric luminosities [erg/s] at each time.

    Returns
    -------
    radius_photosphere : array_like
        Photospheric radius [cm] at each time.
    temperature_photosphere : array_like
        Photospheric temperature [K] at each time.
    """
    # Convert velocity and time to CGS
    v_cgs = v_ejecta * km_cgs         # [cm/s]
    t_cgs = times * day_cgs           # [s]

    # Photosphere radius squared: R² = (v * t)²
    radius_squared = (v_cgs * t_cgs) ** 2

    # Recombination radius squared: R² = L / (4πσT⁴)
    rec_radius_squared = luminosities / (4.0 * pi * sigma_sb_cgs * temperature_floor**4)

    # Use smaller of dynamical and recombination radius
    use_dyn = radius_squared < rec_radius_squared
    radius_photosphere_squared = xp.where(use_dyn, radius_squared, rec_radius_squared)

    # Temperature: T = (L / (4πσR²))^{1/4}, or zero if L == 0
    temperature_photosphere = xp.where(luminosities > 0,
    (luminosities / (4.0 * pi * sigma_sb_cgs * radius_photosphere_squared)) ** 0.25,
        0.0)

    return xp.sqrt(radius_photosphere_squared), temperature_photosphere


@jit
def mosfit_rprocess(m_ejecta, v_ejecta, times,
                    therm_func_a=therm_func_a,
                    therm_func_b=therm_func_b,
                    therm_func_d=therm_func_d):
    """
    Compute the r-process luminosity evolution for kilonova ejecta using a backend-flexible
    version of the MOSFiT r-process heating model.

    Parameters:
    - m_ejecta: float
        Ejecta mass in solar masses.
    - v_ejecta: float
        Ejecta velocity in km/s.
    - times: array
        Observation times in days.
    - therm_func_a, therm_func_b, therm_func_d: callables
        Interpolating functions for heating model coefficients (from Barnes+2016),
        must be backend-compatible (JAX or Numba).

    Returns:
    - luminosities: array
        r-process luminosity values at each input time.
    """

    v_frac_c = v_ejecta / (c_cgs / km_cgs)

    # Evaluate interpolated thermal coefficients
    a = therm_func_a(m_ejecta, v_frac_c)
    b = 2.0 * therm_func_b(m_ejecta, v_frac_c)
    d = therm_func_d(m_ejecta, v_frac_c)

    # Luminosity scaling constant
    L_scale = m_ejecta * M_sun_cgs* 4.0e18 * 0.36

    # Time-dependent heating rate (Korobkin et al. 2012)
    time_sec = times * day_cgs
    arctan_term = (0.5 - (1.0 / pi) * xp.arctan((time_sec - 1.3) / 0.11))**1.3
    decay_term = xp.exp(-a * times)
    log_term = xp.log1p(b * times**d) / (b * times**d)

    luminosities = L_scale * arctan_term * (decay_term + log_term)

    return xp.nan_to_num(luminosities)


@jit
def mosfit_shock_cocoon(m_ejecta, v_ejecta, kappa, cos_theta_cocoon, shocked_fraction, pl_s, t_shock, times):
    """
    Compute the luminosity from a shock-heated cocoon in the BNS model using a backend-flexible formulation.

    Parameters:
    - m_ejecta: float
        Total ejecta mass in solar masses.
    - v_ejecta: float
        Ejecta velocity in km/s.
    - kappa: float
        Opacity of the material.
    - cos_theta_cocoon: float
        Cosine of the cocoon opening angle (range 0 to 1).
    - shocked_fraction: float
        Fraction of the ejecta that is shocked (0 to 1).
    - pl_s: float
        Power-law index of the shocked emission.
    - t_shock: float
        Shock breakout time in seconds.
    - times: array
        Observation times in days.

    Returns:
    - luminosities: array
        Luminosity from the cocoon at each time step.
    """

    # Compute shocked region radius and mass
    shock_radius = c_cgs * t_shock
    mass_shocked = m_ejecta * shocked_fraction
    theta_cocoon = xp.arccos(cos_theta_cocoon)

    # Compute diffusion timescale and optically thin timescale
    diffusion_constant = M_sun_cgs / (4 * pi * c_cgs * km_to_cm)
    tau_diffusion = xp.sqrt(diffusion_constant * kappa * mass_shocked / v_ejecta) / day_to_s
    time_thin = xp.sqrt((c_cgs / km_to_cm) / v_ejecta) * tau_diffusion

    # Luminosity scaling constant
    constant_L0 = (theta_cocoon**2 / 2)**(1 / 3) * (
        mass_shocked * M_sun_cgs * v_ejecta * km_to_cm * shock_radius / (tau_diffusion * day_to_s)**2)

    # Time-dependent luminosity evolution
    decay_term = (times / tau_diffusion)**(-(4 / (pl_s + 2)))
    tanh_term = 0.5 * (1 + xp.tanh(time_thin - times))
    luminosities = constant_L0 * decay_term * tanh_term

    return xp.nan_to_num(luminosities)


@jit
def mosfit_shock_photosphere(m_ejecta, v_ejecta, kappa, shocked_fraction, pl_s, times, luminosities):
    """
    Compute the photospheric radius and temperature for shock-heated ejecta in a BNS system,
    compatible with JAX and Numba via GEMMA's backend abstraction.

    Parameters:
    - m_ejecta: float
        Total ejecta mass in solar masses.
    - v_ejecta: float
        Ejecta velocity in km/s.
    - kappa: float
        Opacity of the material.
    - shocked_fraction: float
        Fraction of the ejecta that is shocked (0 to 1).
    - pl_s: float
        Power-law index for the shocked material's emission.
    - times: array
        Observation times in days.
    - luminosities: array
        Luminosity values corresponding to `times`.

    Returns:
    - radius_photosphere: array
        Photospheric radii at each time (cm).
    - temperature_photosphere: array
        Photospheric temperatures at each time (K).
    """
    day_to_s=day_cgs
    # Shocked ejecta mass
    mass_shocked = m_ejecta * shocked_fraction

    # Diffusion timescale and optically thin timescale
    diffusion_constant = M_sun_cgs / (4 * pi * c_cgs * km_to_cm)
    tau_diffusion = xp.sqrt(diffusion_constant * kappa * mass_shocked / v_ejecta) / day_to_s
    time_thin = xp.sqrt((c_cgs / km_to_cm) / v_ejecta) * tau_diffusion

    # Photosphere velocity and radius
    v_photosphere = v_ejecta * (times / time_thin)**(-2 / (pl_s + 3))
    radius_photosphere = (km_to_cm / day_to_s) * v_photosphere * times

    # Photosphere temperature from Stefan-Boltzmann law
    temperature_photosphere = (luminosities / (4 * pi * sigma_sb_cgs * radius_photosphere**2))**0.25

    return radius_photosphere, temperature_photosphere

@jit
def trapz_like(y, dx):
    """Trapezoidal integration compatible with JAX."""
    return dx * (xp.sum(y) - 0.5 * (y[0] + y[-1]))

@jit
def classify_remnant(total_mass, m_tov, threshold_mass):
    """Remnant classification using lax.switch to avoid Python control flow."""
    conds = [
        total_mass < m_tov,
        total_mass < 1.2 * m_tov,
        total_mass < threshold_mass,
    ]

    def case_0(_):
        return 0.38, 0.15

    def case_1(_):
        Ye = (0.34 - 0.38) / (1.2 * m_tov - m_tov) * (total_mass - m_tov) + 0.38
        vdisk = (0.03 - 0.15) / (threshold_mass - m_tov) * (total_mass - m_tov) + 0.15
        return Ye, vdisk

    def case_2(_):
        Ye = (0.25 - 0.34) / (threshold_mass - 1.2 * m_tov) * (total_mass - 1.2 * m_tov) + 0.34
        vdisk = (0.03 - 0.15) / (threshold_mass - m_tov) * (total_mass - m_tov) + 0.15
        return Ye, vdisk

    def case_3(_):
        return 0.25, 0.03

    idx = xp.select(conds, xp.array([0, 1, 2]), default=3)
    return lax.switch(idx, [case_0, case_1, case_2, case_3], operand=None)

@jit
def mosfit_bns_quantities_jax(mass_1, mass_2,
                               cos_theta_open,
                               m_dyn, m_wind,
                               compactness_1, compactness_2,
                               m_tov, threshold_mass,
                               alpha, kappa_red, kappa_blue):
    mass_ratio = mass_1 / mass_2
    total_mass = mass_1 + mass_2

    red_frac = xp.clip(14.8609 * mass_ratio**2 - 28.6148 * mass_ratio + 13.9597, 0.0, 1.0)

    m_ejecta_red = m_dyn * red_frac
    m_ejecta_blue = m_dyn * (1.0 - red_frac)

    # Compute velocity components
    vdyn_p = -0.219479 * (
        mass_ratio * (1 - 2.67385 * compactness_1) +
        (1 / mass_ratio) * (1 - 2.67385 * compactness_2)
    ) + 0.444836

    vdyn_z = -0.315585 * (
        mass_ratio * (1 - 1.00757 * compactness_1) +
        (1 / mass_ratio) * (1 - 1.00757 * compactness_2)
    ) + 0.63808

    dtheta = 0.01
    theta_full = xp.arange(0.0, pi / 2, dtheta)
    theta_open = xp.arccos(cos_theta_open)

    mask_1 = theta_full < theta_open
    mask_2 = theta_full >= theta_open

    vtheta = xp.sqrt((vdyn_z * xp.cos(theta_full))**2 + (vdyn_p * xp.sin(theta_full))**2)
    atheta = 2.0 * pi * xp.sin(theta_full)

    vtheta_1 = xp.where(mask_1, vtheta, 0.0)
    atheta_1 = xp.where(mask_1, atheta, 0.0)
    vtheta_2 = xp.where(mask_2, vtheta, 0.0)
    atheta_2 = xp.where(mask_2, atheta, 0.0)

    numerator_1 = trapz_like(vtheta_1 * atheta_1, dtheta)
    denominator_1 = trapz_like(atheta_1, dtheta)
    v_blue = numerator_1 / (denominator_1 + 1e-12)

    numerator_2 = trapz_like(vtheta_2 * atheta_2, dtheta)
    denominator_2 = trapz_like(atheta_2, dtheta)
    v_red = numerator_2 / (denominator_2 + 1e-12)

    # Remnant classification
    Ye, vdisk = classify_remnant(total_mass, m_tov, threshold_mass)

    # Adjust blue ejecta if black hole formed
    m_ejecta_blue = xp.where(total_mass > threshold_mass, m_ejecta_blue / alpha, m_ejecta_blue)

    m_ejecta_purple = m_wind
    v_purple = vdisk

    kappa_purple = 2112.0 * Ye**3 - 2238.9 * Ye**2 + 742.35 * Ye - 73.14

    total_m = m_ejecta_red + m_ejecta_blue + m_ejecta_purple
    v_mean = (
        m_ejecta_purple * v_purple +
        v_red * m_ejecta_blue +
        v_blue * m_ejecta_red
    ) / (total_m + 1e-12)

    kappa_mean = (
        m_ejecta_purple * kappa_purple +
        kappa_red * m_ejecta_red +
        kappa_blue * m_ejecta_blue
    ) / (total_m + 1e-12)

    return (m_ejecta_red, m_ejecta_blue, m_ejecta_purple,
            v_red, v_blue, v_purple,
            v_mean, kappa_purple, kappa_mean)


@jit
def trapz_like(y, dx):
    return 0.5 * dx * (y[0] + 2.0 * xp.sum(y[1:-1]) + y[-1])

@jit
def mosfit_bns_quantities_numba(mass_1, mass_2,
                          cos_theta_open,
                          m_dyn, m_wind,
                          compactness_1, compactness_2,
                          m_tov, threshold_mass,
                          alpha, kappa_red, kappa_blue):
    mass_ratio = mass_1 / mass_2
    total_mass = mass_1 + mass_2

    # Polynomial fit for red ejecta fraction, clipped to [0, 1]
    red_frac = 14.8609 * mass_ratio**2 - 28.6148 * mass_ratio + 13.9597
    red_frac = min(max(red_frac, 0.0), 1.0)

    m_ejecta_red = m_dyn * red_frac
    m_ejecta_blue = m_dyn * (1.0 - red_frac)

    # Compute velocity components
    vdyn_p = -0.219479 * (
        (mass_ratio) * (1 - 2.67385 * compactness_1) +
        (1 / mass_ratio) * (1 - 2.67385 * compactness_2)
    ) + 0.444836

    vdyn_z = -0.315585 * (
        (mass_ratio) * (1 - 1.00757 * compactness_1) +
        (1 / mass_ratio) * (1 - 1.00757 * compactness_2)
    ) + 0.63808

    dtheta = 0.01
    theta_full = xp.arange(0.0, pi / 2, dtheta)
    theta_open = xp.arccos(cos_theta_open)

    mask_1 = theta_full < theta_open
    mask_2 = theta_full >= theta_open

    vtheta = xp.sqrt((vdyn_z * xp.cos(theta_full))**2 + (vdyn_p * xp.sin(theta_full))**2)
    atheta = 2.0 * pi * xp.sin(theta_full)

    vtheta_1 = xp.where(mask_1, vtheta, 0.0)
    atheta_1 = xp.where(mask_1, atheta, 0.0)
    vtheta_2 = xp.where(mask_2, vtheta, 0.0)
    atheta_2 = xp.where(mask_2, atheta, 0.0)

    numerator_1 = trapz_like(vtheta_1 * atheta_1, dtheta)
    denominator_1 = trapz_like(atheta_1, dtheta)
    v_blue = numerator_1 / (denominator_1 + 1e-12)

    numerator_2 = trapz_like(vtheta_2 * atheta_2, dtheta)
    denominator_2 = trapz_like(atheta_2, dtheta)
    v_red = numerator_2 / (denominator_2 + 1e-12)

    # Remnant classification and Ye, vdisk
    if total_mass < m_tov:
        Ye, vdisk = 0.38, 0.15
    elif total_mass < 1.2 * m_tov:
        Ye = (0.34 - 0.38) / (1.2 * m_tov - m_tov) * (total_mass - m_tov) + 0.38
        vdisk = (0.03 - 0.15) / (threshold_mass - m_tov) * (total_mass - m_tov) + 0.15
    elif total_mass < threshold_mass:
        Ye = (0.25 - 0.34) / (threshold_mass - 1.2 * m_tov) * (total_mass - 1.2 * m_tov) + 0.34
        vdisk = (0.03 - 0.15) / (threshold_mass - m_tov) * (total_mass - m_tov) + 0.15
    else:
        Ye, vdisk = 0.25, 0.03

    # Adjust blue ejecta if black hole formed
    m_ejecta_blue = xp.where(total_mass > threshold_mass, m_ejecta_blue / alpha, m_ejecta_blue)

    m_ejecta_purple = m_wind
    v_purple = vdisk

    kappa_purple = 2112.0 * Ye**3 - 2238.9 * Ye**2 + 742.35 * Ye - 73.14

    total_m = m_ejecta_red + m_ejecta_blue + m_ejecta_purple
    v_mean = (
        m_ejecta_purple * v_purple +
        v_red * m_ejecta_blue +
        v_blue * m_ejecta_red
    ) / (total_m + 1e-12)

    kappa_mean = (
        m_ejecta_purple * kappa_purple +
        kappa_red * m_ejecta_red +
        kappa_blue * m_ejecta_blue
    ) / (total_m + 1e-12)

    return (m_ejecta_red, m_ejecta_blue, m_ejecta_purple,
            v_red, v_blue, v_purple,
            v_mean, kappa_purple, kappa_mean)



@jit
def mosfit_bns_magnetar(
    mass_1, mass_2,
    period_spin, magnetic_field,
    albedo, pair_multiplicity,
    v_ejecta_mean, epsilon_therm,
    m_dyn, m_disk,
    m_tov, R16,
    times):
    """
    Compute magnetar-powered luminosity following Gompertz et al. 2023,
    based on spin-down energy and collapse energy.

    Parameters
    ----------
    mass_1, mass_2 : float
        NS masses.
    period_spin : float
        Initial spin period in ms.
    magnetic_field : float
        Magnetic field in Gauss.
    albedo : float
        Reprocessing efficiency.
    pair_multiplicity : float
        Pair multiplicity in the nebula.
    v_ejecta_mean : float
        Ejecta velocity in km/s.
    epsilon_therm : float
        Thermalization efficiency.
    m_dyn, m_disk, m_tov, R16 : float
        Ejecta masses, TOV limit, and neutron star radius in km.
    times : array
        Time array in days.

    Returns
    -------
    luminosities : array
        Spin-down powered luminosities (erg/s).
    """

    mass_remnant = mass_1 + mass_2 - m_dyn - m_disk
    moment_of_inertia = 1.3e45 * (mass_remnant / 1.4)**1.5
    inertia = (2 / 5) * mass_remnant * M_sun_cgs * (R16 * km_cgs)**2

    omega_crit = xp.sqrt(
        G_cgs * (xp.maximum(mass_remnant - m_tov, 0.0)) * M_sun_cgs / (R16 * km_cgs)**3)

    period_crit = 2.0 * pi / (omega_crit + 1e-30) * 1000.0  # ms

    rotational_energy = 1.0e53 * (inertia / moment_of_inertia) * \
        (mass_remnant / 2.3)**1.5 * (period_spin / 0.7)**-2

    collapse_energy = 1.0e53 * (inertia / moment_of_inertia) * \
        (mass_remnant / 2.3)**1.5 * (period_crit / 0.7)**-2

    L0 = 7.0e48 * (inertia / moment_of_inertia) * magnetic_field**2 * \
         (period_spin / 0.7)**-4 * (mass_remnant / 2.3)**1.5 * (R16 / 12.0)**2

    spin_down_time = rotational_energy / (L0 + 1e-30)
    available_energy = rotational_energy - collapse_energy

    t_sec = times * day_cgs
    L = L0 / (1.0 + t_sec / (spin_down_time + 1e-30))

    # Compute cumulative energy radiated (integral approximation)
    dt = t_sec[1] - t_sec[0]
    radiated_energy = xp.cumsum((L[:-1] + L[1:]) * 0.5 * dt)
    radiated_energy = xp.concatenate([xp.array([0.0]), radiated_energy])

    # Zero out luminosities beyond available energy
    mask = radiated_energy > available_energy
    L = xp.where(mask, 0.0, L)

    # Pair-opacity correction
    v_factor = v_ejecta_mean * km_cgs / (0.3 * c_cgs)
    t_life_over_t0 = 0.6 / (1.0 - albedo + 1e-12) * (pair_multiplicity / 0.1)**0.5 * v_factor**0.5
    t_life_over_t = t_life_over_t0 * (L / 1e45)**0.5 * times**-0.5
    L_final = L * epsilon_therm / (1.0 + t_life_over_t)

    return xp.nan_to_num(L_final)
