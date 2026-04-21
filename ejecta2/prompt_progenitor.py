# -*- coding: utf-8 -*-
"""
PromptX wrapper — JAX-accelerated, fully self-contained.
"""

from scripts.jet import Jet
from scripts.helper import *
from scripts.helper import _int_energy_1d, _interp_1d, band_broadcast
from scripts.const import *
from scripts.units_conversion import *
from scripts.functions import calculate_ns_compactness, calculate_baryonic_mass, calculate_r_isco
from scripts.bns import calculate_bns_disk_mass_kruger, calculate_bns_dynamical_mass_kruger

import jax.numpy as jnp
import numpy as np
import math

# Optional components
try:
    from scripts.wind_enhanced import Wind
except ImportError:
    Wind = None
try:
    from scripts.magnetar_enhanced import Magnetar
except ImportError:
    Magnetar = None

EV_TO_ERG = 1.602176634e-12


class Prompt:
    def __init__(
        self,
        components=("jet",),
        sample_gw_parameters=False,
        gw_param_mode="chirp_mass",
        j_struct=2,
        eta_jet=1e-4,
        frac=0.1,
        phi_los_deg=0.0,
        theta_c_deg=5.0,
        theta_cut_deg=35.0,
        inclination_key="theta_jn",
        default_theta_los_deg=0.0,
        use_disk_mass_mapping=True,
        output='luminosity',
        distance_key="luminosity_distance",
        default_distance_mpc=40.0,
        sample_distance=False
    ):
        self.use_disk_mass_mapping = use_disk_mass_mapping
        self.components = list(components)
        self.sample_gw_parameters = bool(sample_gw_parameters)
        self.gw_param_mode = gw_param_mode
        self.sample_distance = bool(sample_distance)
        self.output = output

        self.theta_c = float(np.deg2rad(theta_c_deg))
        self.theta_cut = float(np.deg2rad(theta_cut_deg))
        self.eta = float(eta_jet)
        self.frac = float(frac)
        self.phi_los = float(np.deg2rad(phi_los_deg))
        self.inclination_key = inclination_key
        self.default_theta_los_deg = float(default_theta_los_deg)
        self.distance_key = distance_key
        self.default_distance_mpc = float(default_distance_mpc)

        self.grid_params = {"n_theta": 500, "n_phi": 100}

        structs = {"tophat": 1, "gaussian": 2, "powerlaw": 3}
        if j_struct is None:
            self.j_idx = 1
        elif isinstance(j_struct, str):
            if j_struct not in structs:
                raise ValueError("Pick a valid jet profile: tophat, gaussian, powerlaw")
            self.j_idx = structs[j_struct]
        elif isinstance(j_struct, (int, np.integer)):
            if j_struct not in (1, 2, 3):
                raise ValueError("j_struct int must be 1, 2, or 3")
            self.j_idx = int(j_struct)
        else:
            raise TypeError("j_struct must be None, str, or int")

        self.parameter_names = []
        self.parameter_bounds = {}

        component_suffix_map = {"jet": "jet", "wind": "wind"}
        jet_base_params = []
        wind_base_params = [
            ("gamma", (2.0, 300.0)), ("theta_cut", (0.2, 90.0)),
            ("E_iso", (1e45, 1e55)), ("collapse", (0.0, 1.0)),
            ("P_0", (0.5e-3, 50e-3)), ("M", (1.0, 3.0)),
            ("B_p", (1e13, 1e16)), ("R", (8e5, 1.6e6)),
            ("eps", (1e-6, 1e-2)), ("kappa", (0.1, 20.0)),
            ("m_ejecta_dyn", (1e-4, 0.1)), ("v_ejecta_dyn", (0.05, 0.4)),
        ]

        for comp in self.components:
            suffix = component_suffix_map[comp]
            base_params = jet_base_params if comp == "jet" else wind_base_params
            for base_name, bounds in base_params:
                pname = f"{base_name}_{suffix}"
                self.parameter_names.append(pname)
                self.parameter_bounds[pname] = bounds

        if not self.sample_gw_parameters:
            self.parameter_names.append("theta_los")
            self.parameter_bounds["theta_los"] = (0.0, 90.0)
        else:
            self.parameter_names.append(self.inclination_key)
            self.parameter_bounds[self.inclination_key] = (0.0, np.pi)
            if self.gw_param_mode == "mass":
                for name, bounds in [("mass_1", (0.8, 3.0)), ("mass_2", (0.8, 3.0)),
                                     ("lambda_2", (0.0, 5000.0))]:
                    self.parameter_names.append(name)
                    self.parameter_bounds[name] = bounds
            elif self.gw_param_mode == "chirp_mass":
                for name, bounds in [("chirp_mass", (0.8, 2.0)), ("mass_ratio", (0.05, 1.0)),
                                     ("lambda_2", (0.0, 5000.0))]:
                    self.parameter_names.append(name)
                    self.parameter_bounds[name] = bounds

        self.params = None
        self.jet_model = None
        self.wind_model = None
        self.total_X = None
        self.t = None

        if ("jet" in self.components) and (not self.use_disk_mass_mapping):
            self.parameter_names.append("E_iso_jet")
            self.parameter_bounds["E_iso_jet"] = (1e45, 1e55)

        if self.output not in ("luminosity", "flux"):
            raise ValueError("output must be 'luminosity' or 'flux'")
        if self.output == "flux" and self.sample_distance:
            self.parameter_names.append(self.distance_key)
            self.parameter_bounds[self.distance_key] = (0.1, 2000.0)

    # ── Public API ─────────────────────────────────────────────────────
    def bounds_check(self, params):
        for name in self.parameter_names:
            if name not in params:
                continue
            lo, hi = self.parameter_bounds[name]
            v = float(params[name])
            if not (lo <= v <= hi):
                raise ValueError(f"[Bound check failed] {name}={v} not in [{lo},{hi}]")

    def update_model(self, params, check_bounds=False, dry_run=False, verbose=True):
        if verbose:
            print('Updating model...')
        if check_bounds:
            self.bounds_check(params)
        self.params = params
        if dry_run:
            theta_los = self._get_theta_los(params)
            E_iso_jet = self._derive_Eiso_jet(params, self.theta_c)
            return {"theta_los": theta_los, "E_iso_jet": E_iso_jet}
        self.generate_light(params)

    # ── Core model evaluation ──────────────────────────────────────────
    def generate_light(self, params):
        theta_los = self._get_theta_los(params)

        jet_L = self.generate_jet_lc(params, theta_los) if "jet" in self.components else None
        wind_L = self.generate_wind_lc(params, theta_los) if "wind" in self.components else None

        if (jet_L is not None) and (wind_L is not None):
            self.total_X = jet_L.L_X_tot + wind_L.L_X_tot
            self.total_gamma = jet_L.L_gamma_tot
            self.t = jet_L.t
        elif jet_L is not None:
            self.total_X = jet_L.L_X_tot
            self.total_gamma = jet_L.L_gamma_tot
            self.t = jet_L.t
        elif wind_L is not None:
            self.total_X = wind_L.L_X_tot
            self.t = wind_L.engine.t
        else:
            raise RuntimeError("No components enabled")

        if self.output == 'flux':
            self.total_X = self._to_flux(self.total_X, params)
            if hasattr(self, 'total_gamma'):
                self.total_gamma = self._to_flux(self.total_gamma, params)

    def generate_jet_lc(self, params, theta_los_rad):
        E_iso_jet = self._derive_Eiso_jet(params, self.theta_c)
        print('eisojet val', E_iso_jet)
        self.E_iso_jet = E_iso_jet

        jet_model = Jet(
            E_iso=E_iso_jet, eps0=E_iso_jet,
            n_theta=self.grid_params["n_theta"],
            n_phi=self.grid_params["n_phi"],
            theta_c=self.theta_c, theta_cut=self.theta_cut,
            jet_struct=self.j_idx,
        )
        jet_model.define_structure(
            eps0=jet_model.eps[0][0], E_iso=E_iso_jet, jet_struct=self.j_idx)
        jet_model.create_obs_grid(amati_a=0.41, amati_b=0.83)
        jet_model.observer(theta_los=float(theta_los_rad), phi_los=float(self.phi_los))
        print('Jet Constructed Successfully...')

        self.jet_model = jet_model
        return jet_model

    def generate_wind_lc(self, params, theta_los_rad):
        if Wind is None:
            raise ImportError("wind_enhanced module not available")
        gamma_wind = float(params["gamma_wind"])
        theta_cut = float(np.deg2rad(float(params["theta_cut_wind"])))
        wind = Wind(g0=gamma_wind, n_theta=self.grid_params["n_theta"],
                    n_phi=self.grid_params["n_phi"], theta_cut=theta_cut)
        wind.observer(theta_los=float(theta_los_rad), phi_los=float(self.phi_los))
        wind.E_iso_wind = float(params["E_iso_wind"])
        wind.collapse = bool(float(params["collapse_wind"]) > 0.5)
        self.wind_model = wind
        return wind

    # ── Helpers ─────────────────────────────────────────────────────────
    def _get_theta_los(self, params):
        if self.sample_gw_parameters:
            return float(params[self.inclination_key])
        if "theta_los" in params:
            return float(np.deg2rad(float(params["theta_los"])))
        return float(np.deg2rad(self.default_theta_los_deg))

    def _extract_masses(self, params):
        if "mass_1" in params and "mass_2" in params:
            return float(params["mass_1"]), float(params["mass_2"])
        if self.gw_param_mode == "mass":
            raise KeyError("mass_1 and mass_2 required")
        raise ValueError("Provide mass_1/mass_2 or implement chirp_mass conversion")

    def _derive_Eiso_jet(self, params, theta_c_rad):
        """Derive E_iso from disk mass mapping (all scalar math)."""
        if not self.use_disk_mass_mapping:
            return float(params["E_iso_jet"])

        m1, m2 = self._extract_masses(params)
        lambda_2 = float(params["lambda_2"])

        C2 = float(calculate_ns_compactness(lambda_2, True, systematics_fraction=0.0))
        m_disk = float(calculate_bns_disk_mass_kruger(m2, C2))
        self.mdisk = m_disk

        if not math.isfinite(self.mdisk) or self.mdisk <= 0.0:
            raise ValueError(f"Rejected: m_disk={self.mdisk}")

        q = max(1e-6, m1 / m2)
        xi = 0.18 + 0.11 / (1.0 + math.exp(1.5 * (q - 3.0)))
        E_k = 0.5 * (1.0 - self.frac) * (1.0 - xi) * (self.mdisk * M_sun_cgs) * self.eta * (c_cgs ** 2)
        denom = max(1e-12, 1.0 - math.cos(theta_c_rad))
        return E_k / denom

    def _get_distance_cm(self, params):
        d_mpc = float(params.get(self.distance_key, self.default_distance_mpc))
        return d_mpc * Mpc_in_cm

    def _to_flux(self, L, params):
        D = self._get_distance_cm(params)
        return L / (4.0 * math.pi * D * D)

    def check_print(self):
        print('Object OK!')
