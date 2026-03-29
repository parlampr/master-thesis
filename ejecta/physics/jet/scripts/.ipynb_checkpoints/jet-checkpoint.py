# ============================================================================= #
#                  ____  ____   __   _  _  ____  ____  _  _                     #
#                 (  _ \(  _ \ /  \ ( \/ )(  _ \(_  _)( \/ )                    #
#                  ) __/ )   /(  O )/ \/ \ ) __/  )(   )  (                     #
#                 (__)  (__\_) \__/ \_)(_/(__)   (__) (_/\_)                    #
#                                                                               #
# ============================================================================= #
#   PromptX - Prompt X-ray emission modeling of relativistic outflows           #
#   Version 1.0                                                                 #
#   Author: Connery Chen, Yihan Wang, and Bing Zhang                            #
#   License: MIT                                                                #
# ============================================================================= # 

from numpy import int_, newaxis
from .helper import *
from .helper import _int_energy_1d, _interp_1d, band_broadcast
from .const import *
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

EV_TO_ERG = 1.602176634e-12

class Jet:
    """
    Represents a relativistic jet launched by a central engine, characterized
    by its energy and Lorentz factor structure as a function of polar angle.
    """
    def __init__(self, n_theta=500, n_phi=100, g0=200, E_iso=1e53, eps0=1e53, theta_c=np.pi/2, k=0, theta_cut=np.pi/2, jet_struct=0):
        """
        Initializes the jet model by setting up the grid, defining energy and Lorentz factor profiles,
        and normalizing the energy distribution.

        Parameters:
            n_theta (int): Number of polar (theta) grid points.
            n_phi (int): Number of azimuthal (phi) grid points.
            g0 (float): Lorentz factor normalization.
            E_iso (float): On-axis isotropic equivalent energy for normalization.
            eps0 (float): On-axis energy per solid angle.
            theta_c (float): Core angle for the jet.
            theta_cut (float): Cutoff angle for the jet structure.
            jet_struct (str or function): Structure type, either 'gaussian', 'powerlaw', or a custom function.
            
        Initializes the following attributes:
            theta_grid (ndarray): 2D grid of theta values for the jet.
            phi_grid (ndarray): 2D grid of phi values for the jet.
            theta (ndarray): 1D array of cell-centered theta values.
            phi (ndarray): 1D array of cell-centered phi values.
            dOmega (ndarray): Differential solid angle for each grid cell.
            theta_c (float): Core angle of the jet.
            theta_cut (float): Cutoff angle for the wind structure.
            eps (ndarray): Energy per solid angle profile of the jet.
            g (ndarray): Lorentz factor profile of the jet.
        """

        # Define the bounds for theta (polar angle) and phi (azimuthal angle)
        theta_bounds = [0, np.pi]
        phi_bounds = [0, 2 * np.pi]
        
        # Generate a grid for theta and phi, based on the specified grid points
        self.theta_grid, self.phi_grid = coord_grid(n_theta, n_phi, theta_bounds, phi_bounds)
        
        # Compute cell-centered theta and phi values
        self.theta = 0.25 * (self.theta_grid[:-1, :-1] + self.theta_grid[1:, :-1] +
                            self.theta_grid[:-1, 1:] + self.theta_grid[1:, 1:])
        self.phi   = 0.25 * (self.phi_grid[:-1, :-1] + self.phi_grid[1:, :-1] +
                            self.phi_grid[:-1, 1:] + self.phi_grid[1:, 1:])

        # Compute the differential solid angle (dOmega) for each grid cell
        dtheta = np.gradient(self.theta, axis=0)
        dphi = np.gradient(self.phi, axis=1)
        self.dOmega = np.sin(self.theta) * dtheta * dphi

        # Set the core and cutoff angle for the jet
        self.theta_c = theta_c
        self.theta_cut = theta_cut
        # only used for PL profile
        self.k = k

        # Define Lorentz factor and energy per solid angle profiles
        self.define_structure(eps0, E_iso, jet_struct)

        # Normalize
        self.normalize(self.E_iso)

    def define_structure(self, eps0, E_iso, jet_struct):
        """
        Defines the structure of the wind's energy and Lorentz factor profiles based on the specified profile type.

        Parameters:
        g0 (float): The initial Lorentz factor normalization.
        eps0 (float): The initial energy per solid angle (before applying structure).
        E_iso (float): The isotropic-equivalent energy used for the Gaussian and power-law structures.
        struct (int or function): The structure type, where:
            - 1: Tophat
            - 2: Gaussian
            - 3: Power-law
            - function: A custom piecewise function to define eps and gamma.

        The function updates the `self.eps` and `self.g` attributes based on the selected structure.
        """

        self.eps0 = eps0  
        self.E_iso = E_iso
        self.struct = jet_struct

        struct_map = {1: 'tophat', 2: 'gaussian', 3: 'powerlaw'}
        if callable(self.struct):
            struct = self.struct
        elif self.struct in struct_map:
            struct = struct_map[self.struct]
        elif isinstance(self.struct, str) and self.struct.lower() in struct_map.values():
            struct = self.struct.lower()
        else:
            raise ValueError(f"Invalid structure type: {self.struct}")

        cutoff = self.theta_cut

        self.eps = eps_grid(self.eps0, self.theta, self.phi, theta_c=self.theta_c, k=self.k, struct=struct, cutoff=cutoff)
        E_iso_profile = eps_grid(self.E_iso, self.theta, self.phi, theta_c=self.theta_c, k=self.k, struct=struct, cutoff=cutoff)
        # self.g = gamma_grid(self.g0, self.theta, k=k, struct=struct, cutoff=cutoff)

        # self.g = gamma_grid(self.g0 , self.theta, k=0, struct='powerlaw')
        self.g = lg11(E_iso_profile)

    def normalize(self, E_iso):
        """
        Normalizes the jet energy profile to match the given isotropic-equivalent energy.

        This method scales the jet's energy profile (per solid angle) such that the observed on-axis 
        isotropic-equivalent energy equals a given value.

        Parameters:
        E_iso (float): The target isotropic-equivalent energy to normalize to.
        """

        # Compute the isotropic equivalent energy per solid angle
        e_iso_grid_N = e_iso_grid(self.theta, self.phi, 0, self.g, self.eps, self.theta_cut, self.dOmega)
        e_iso_grid_S = e_iso_grid(self.theta, self.phi, np.pi, self.g, self.eps, self.theta_cut, self.dOmega)
        self.e_iso_grid = e_iso_grid_N + e_iso_grid_S
        # Compute the normalization factor
        A = E_iso / self.e_iso_grid[0]

        # Apply the normalization
        self.eps *= A
        
        # Recalculate E_iso per grid
        e_iso_grid_N = e_iso_grid(self.theta, self.phi, 0, self.g, self.eps, self.theta_cut, self.dOmega)
        e_iso_grid_S = e_iso_grid(self.theta, self.phi, np.pi, self.g, self.eps, self.theta_cut, self.dOmega)
        self.e_iso_grid = e_iso_grid_N + e_iso_grid_S

    def create_obs_grid(self, amati_a=0.41, amati_b=0.83):
        """
        Generate observer-frame spectral and temporal grids for gamma-ray and X-ray bands.

        This method computes the on-grid energy array, photon number spectrum, time array,
        luminosity light curves, and integrated fluences for two energy ranges:
        - Gamma-rays: 10 keV to 1000 keV
        - X-rays: 0.3 keV to 10 keV

        The Amati relation, E_p = 1e5 * 10**(amati_a * np.log10(e_iso_grid / 1e51) + amati_b)
        is used to set the spectral peak energy based on the input `amati_a` and `amati_b`.

        Args:
            amati_a (float, optional): Slope of the Amati relation to use when determining
                the rest-frame peak energy. 
            amati_b (float, optional): Intercept of the Amati relation to use when determining
                the rest-frame peak energy. 
        """

        # Calculate on-grid spectrum and light curve for gamma rays (10e3 - 1000e3 eV)
        self.E, self.EN_E, self.t, self.L_gamma, self.S_gamma = obs_grid(self.eps, self.e_iso_grid, amati_a=amati_a, amati_b=amati_b, e_1=10e3, e_2=1000e3)

        # Calculate on-grid spectrum and light curve for X-rays (0.3e3 - 10e3 eV)
        _, _, _, self.L_X, self.S_X = obs_grid(self.eps, self.e_iso_grid, amati_a=amati_a, amati_b=amati_b, e_1=0.3e3, e_2=10e3)

    def observer(self, theta_los=0, phi_los=0):
        """
        Calculates observer-frame properties of the jet at a given line of sight.

        Parameters:
            theta_los (float): Line-of-sight polar angle (in radians). 
            phi_los (float): Line-of-sight azimuthal angle (in radians). 
        """
        # Find grid coordinates corresponding to line of sight (LoS)
        los_coord = nearest_coord(self.theta, self.phi, theta_los, phi_los)

        # Compute ratio of Doppler factors
        D_on = doppf(self.g, 0)
        D_off = doppf(self.g, angular_d(self.theta[los_coord[0], los_coord[1]], self.theta, self.phi[los_coord[0], los_coord[1]], self.phi))
        self.R_D = D_off / D_on

        # EATS
        beta = gamma2beta(self.g)
        theta_obs = angular_d(theta_los, self.theta, phi_los, self.phi)
        t_lab = self.t[np.newaxis, np.newaxis, :]
        R_em = c * t_lab * beta[..., np.newaxis] / (1 - beta[..., np.newaxis])
        self.t_obs = self.t + R_em / c * (1 - np.cos(theta_obs)[..., np.newaxis])
        
        EN_E_per_sa_obs = self.EN_E * self.R_D[..., np.newaxis]**3
        L_gamma_per_sa_obs = self.L_gamma * self.R_D[..., np.newaxis]**4
        L_X_per_sa_obs = self.L_X * self.R_D[..., np.newaxis]**4

        # Adjust spectrum and LC by Doppler ratio
        self.EN_E_obs = EN_E_per_sa_obs * self.dOmega[..., np.newaxis]
        self.L_gamma_obs = L_gamma_per_sa_obs * self.dOmega[..., np.newaxis]
        self.L_X_obs = L_X_per_sa_obs * self.dOmega[..., np.newaxis]

        # Sum the spectra over emitting regions
        self.spec_tot = 4 * np.pi * np.sum(self.EN_E_obs, axis=(0, 1))

        # Interpolate the light curves for gamma-ray and X-ray emissions
        self.t, self.L_gamma_tot = interp_lc(self.t_obs, self.L_gamma_obs)
        _, self.L_X_tot = interp_lc(self.t_obs, self.L_X_obs)

        # Weight by solid angle
        weight = np.sum(self.dOmega[..., np.newaxis], axis=(0, 1))
        self.L_gamma_tot /= weight
        self.L_X_tot /= weight
        self.spec_tot /= weight

        # Calculate observed energy per solid angle
        self.eps_bar_gamma = int_spec(self.E, self.spec_tot, E_min=10e3, E_max=1000e3)

        # Calculate isotropic-equivalent properties
        self.L_gamma_tot *= 4 * np.pi
        self.L_X_tot *= 4 * np.pi
        self.E_iso_obs = self.eps_bar_gamma
        self.L_iso_obs = int_lc(self.t, self.L_gamma_tot)

        # print('Spectrum integral:', int_spec(self.E, self.spec_tot, E_min=10e3, E_max=1000e3))
        # print('Light curve integral (gamma):', int_lc(self.t, self.L_gamma_tot))
        # print('Light curve integral (X):', int_lc(self.t, self.L_X_tot))
        # print('E_peak:', self.E[np.argmax(self.E * self.spec_tot)], 'eV')
        # print('L_gamma_peak:', np.max(self.L_gamma_tot))
        # print('t_peak', self.t[np.argmax(self.L_gamma_tot)])

        self.S_prime = self.S_gamma * self.R_D
        self.S_prime3 = self.S_gamma * self.R_D**3

    def refine_grid(self, theta_los, phi_los):
        """
        Rotate the grid so that the Doppler-brightest spot aligns with the new pole.
        """
        # --- Step 1: Compute R_D on cell centers ---
        los_coord = nearest_coord(self.theta, self.phi, theta_los, phi_los)
        theta_los_val = self.theta[los_coord[0], los_coord[1]]
        phi_los_val   = self.phi[los_coord[0], los_coord[1]]

        D_on = doppf(self.g, 0)
        D_off = doppf(self.g, angular_d(theta_los_val, self.theta, phi_los_val, self.phi))
        R_D = D_off / D_on

        # --- Step 2: Find max R_D location ---
        imax, jmax = np.unravel_index(np.argmax(R_D), R_D.shape)
        theta_peak = self.theta[imax, jmax]
        phi_peak = self.phi[imax, jmax]

        # --- Step 3: Rotate edges (theta_grid, phi_grid) ---
        x = np.sin(self.theta_grid) * np.cos(self.phi_grid)
        y = np.sin(self.theta_grid) * np.sin(self.phi_grid)
        z = np.cos(self.theta_grid)
        xyz = np.stack([x, y, z], axis=-1)

        target_vec = np.array([
            np.sin(theta_peak) * np.cos(phi_peak),
            np.sin(theta_peak) * np.sin(phi_peak),
            np.cos(theta_peak)
        ])
        north_pole = np.array([0.0, 0.0, 1.0])

        rot_axis = np.cross(north_pole, target_vec)
        axis_norm = np.linalg.norm(rot_axis)
        if axis_norm < 1e-12:
            xyz_rot = xyz
        else:
            rot_axis /= axis_norm
            angle = np.arccos(np.clip(np.dot(north_pole, target_vec), -1.0, 1.0))
            K = np.array([[0, -rot_axis[2], rot_axis[1]],
                        [rot_axis[2], 0, -rot_axis[0]],
                        [-rot_axis[1], rot_axis[0], 0]])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            xyz_rot = xyz @ R.T

        # --- Step 4: Convert back to spherical ---
        x_rot, y_rot, z_rot = xyz_rot[...,0], xyz_rot[...,1], xyz_rot[...,2]
        self.theta_grid = np.arccos(np.clip(z_rot, -1.0, 1.0))
        self.phi_grid = np.arctan2(y_rot, x_rot) % (2*np.pi)

        # --- Step 5: Compute cell-centered theta/phi correctly ---
        xyz_cells = 0.25 * (xyz_rot[:-1, :-1] + xyz_rot[1:, :-1] +
                            xyz_rot[:-1, 1:] + xyz_rot[1:, 1:])
        x_c, y_c, z_c = xyz_cells[...,0], xyz_cells[...,1], xyz_cells[...,2]
        self.theta = np.arccos(np.clip(z_c, -1.0, 1.0))
        self.phi = np.arctan2(y_c, x_c) % (2*np.pi)

        # --- Step 6: Compute dOmega from rotated edges ---
        dtheta = self.theta_grid[1:, :-1] - self.theta_grid[:-1, :-1]
        dphi   = self.phi_grid[:-1, 1:] - self.phi_grid[:-1, :-1]
        self.dOmega = np.sin(0.25*(self.theta_grid[:-1, :-1] + self.theta_grid[1:, 1:])) * dtheta * dphi

        
    def time_resolved_spectrum(self, band='gamma', E_band=None, t_common=None, d_l_cm=None):
        """
        Memory-safe, detector-agnostic, time-resolved photon spectrum N(E,t)
    
        Returns flux if distance from earth is given, else just the spectrum
        """
        # ---- choose which LC to use (only for normalization/driver), not to restrict energy range
        if E_band is not None:
            E1, E2 = E_band
            use_X = (E2 <= 10e3)  # crude hint; not critical
        elif band == 'X':
            E1, E2 = 0.3e3, 10e3
            use_X = True
        else:
            E1, E2 = 10e3, 1e6
            use_X = False
    
        # default observer-frame time grid
        if t_common is None:
            t_common = np.geomspace(1e-3, 1e6, 800)
    
        # ... keep your band selection and t_common as-is ...

        E = self.E.astype(float)
        nE = E.size
        t_out = t_common
        N_sum = np.zeros((nE, t_out.size), dtype=float)   # accumulator over solid angle
        E_inv = 1.0 / E
    
        # pick per-patch source LC cube (still in source convention; we apply D^4 below)
        L_src = self.L_X if use_X else self.L_gamma      # (nθ, nφ, nt)
    
        n_th, n_ph = self.theta.shape
        for i in range(n_th):
            for j in range(n_ph):
                if not np.isfinite(self.dOmega[i, j]) or self.dOmega[i, j] <= 0.0:
                    continue
    
                # --- inside the double loop over (i,j) ---

                D = float(self.R_D[i, j])
                
                # 1) νFν normalizer in *source* band (DO NOT shift edges)
                #    S_src = ∫ EN_src(E) * E dE over [E1, E2]
                S_src_nufnu = int_spec(self.E, self.EN_E[i, j, :], E_min=E1, E_max=E2)
                S_obs_nufnu = (D**3) * S_src_nufnu
                if not np.isfinite(S_obs_nufnu) or S_obs_nufnu <= 0.0:
                    continue
                
                # 2) Observer-frame energy spectrum without shifting E (mimic observer()):
                #    EN_obs(E) = D^3 * EN_src(E)  (NO E/D here)
                EN_obs = (D**3) * self.EN_E[i, j, :]
                
                # 3) Unit νFν-normalized shape φ(E) so that ∫ E φ(E) dE = 1 in [E1, E2]
                phi_ij = EN_obs / S_obs_nufnu  # (nE,)
                
                # 4) LC (per solid angle) with D^4 as you already have
                tij = self.t_obs[i, j, :]
                Lij = (self.L_X if use_X else self.L_gamma)[i, j, :] * (D**4)
                valid = np.isfinite(tij) & np.isfinite(Lij) & (tij > 0)
                if valid.sum() < 2:
                    continue
                L_interp = np.interp(t_out, tij[valid], Lij[valid], left=0.0, right=0.0)
                
                # 5) Accumulate photon spectrum: N_ij(E,t) = [φ(E)/E] * L_ij(t), then × dΩ
                contrib = (phi_ij * (1.0 / self.E))[:, None] * L_interp[None, :] * self.dOmega[i, j]
                N_sum += contrib

    
        # --- iso-equivalent average over solid angle ---
        weight = np.sum(self.dOmega)
        N_iso  = (4.0 * np.pi / weight) * N_sum   # [ph s^-1 eV^-1]
    
        if d_l_cm is not None:
            N_iso = N_iso / (4.0 * np.pi * d_l_cm**2)  # flux
    
        return E, t_out, N_iso



    def time_energy_map(self, quantity='nuFnu', band='gamma', E_band=None, t_common=None, d_l_cm=None):
        """
        Build a time–energy map in your preferred units.
    
        quantity:
          'N'     -> N(E,t)                     [ph s^-1 eV^-1] or [ph s^-1 cm^-2 eV^-1]
          'EN'    -> E*N(E,t)                   [erg s^-1 eV^-1] or [erg s^-1 cm^-2 eV^-1]
          'nuFnu' -> E^2*N(E,t) (per log E)     [erg s^-1]       or [erg s^-1 cm^-2]
        """
        E, t, N = self.time_resolved_spectrum(band=band, E_band=E_band, t_common=t_common, d_l_cm=d_l_cm)
    
        if quantity == 'N':
            Z = N
            units = r'ph s$^{-1}$ eV$^{-1}$' if d_l_cm is None else r'ph s$^{-1}$ cm$^{-2}$ eV$^{-1}$'
        elif quantity == 'EN':
            Z = (E[:, None] * N) * EV_TO_ERG
            units = r'erg s$^{-1}$ eV$^{-1}$' if d_l_cm is None else r'erg s$^{-1}$ cm$^{-2}$ eV$^{-1}$'
        elif quantity == 'nuFnu':
            Z = (E[:, None]**2 * N) * EV_TO_ERG
            units = r'E**2*N (erg s$^{-1}$)' if d_l_cm is None else r'erg s$^{-1}$ cm$^{-2}$'
        else:
            raise ValueError("quantity must be 'N', 'EN', or 'nuFnu'.")
    
        return E, t, Z, units

    
    def plot_time_resolved_cmap(self, quantity='nuFnu', band='X', E_band=None, d_l_cm=None,
                                t_common=None, energy_unit='keV',
                                cmap='inferno', log_norm=True, vmin=None, vmax=None,
                                title=None):
        """
        Colormap of the chosen time–energy quantity.
        quantity: 'N' | 'EN' | 'nuFnu'  (see time_energy_map docstring)
        """
        E, t, Z, units = self.time_energy_map(quantity=quantity, band=band, E_band=E_band,
                                              t_common=t_common, d_l_cm=d_l_cm)
    
        # axis units
        scale = {'eV':1.0, 'keV':1e-3, 'MeV':1e-6}.get(energy_unit, 1e-3)
        E_plot = E * scale
    
        # helper to build bin edges for log grids
        def edges_from_centers(x):
            x = np.asarray(x)
            edges = np.empty(x.size + 1)
            edges[1:-1] = np.sqrt(x[:-1] * x[1:])
            edges[0]  = x[0]**2 / edges[1]
            edges[-1] = x[-1]**2 / edges[-2]
            return edges
    
        E_edges = edges_from_centers(E_plot)
        t_edges = edges_from_centers(t)
    
        # choose normalization
        if log_norm:
            Zplot = np.where((Z > 0) & np.isfinite(Z), Z, np.nan)
            if vmin is None: vmin = np.nanpercentile(Zplot, 5)
            if vmax is None: vmax = np.nanpercentile(Zplot, 99)
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            Zplot = Z
            norm = None
    
        fig, ax = plt.subplots(figsize=(7, 4.2))
        pm = ax.pcolormesh(t_edges, E_edges, Zplot, shading='auto', cmap=cmap, norm=norm)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(f'Energy [{energy_unit}]')
        cbar = fig.colorbar(pm, ax=ax, pad=0.02)
        cbar.set_label(units)
        if title is None:
            title = {'N':'Photon spectrum', 'EN':'Energy density (per eV)', 'nuFnu':'Time-dep Spectrum (per log E)'}[quantity]
            title += ' (no detector)'
        ax.set_title(title)
        fig.tight_layout()
        return fig, ax, E, t, Z

    def time_resolved_spectrum_full(self,
                               band='X',          # 'X' or 'gamma' (chooses driver LC only)
                               E_band=None,           # [E1,E2] eV; if None use PromptX defaults per 'band'
                               t_common=None,
                               d_l_cm=None,           # if set, return flux (divide by 4π d_L^2)
                               z=0.0,                 # cosmological redshift (ignored in 'promptx' mode)
                               mode='promptx'         # 'promptx' | 'energy'
                               ):
        """
        N(E,t) time-resolved spectrum.
    
        mode='promptx'  -> mimic PromptX observer(): NO energy shift; spectra × D^3, LCs × D^4; νFν-in-linear-E normalization.
        mode='energy'   -> physics/cross-band: DO energy shift E_obs = D*E_src/(1+z); build shapes in observer frame;
                           band power is energy-conserving ∫ E N dE.
        """
    
        # ---------------- band and time grid ----------------
        if E_band is not None:
            E1, E2 = E_band
            use_X = (E2 <= 10e3)
        elif band == 'X':
            E1, E2 = 0.3e3, 10e3  # PromptX X band
            use_X = True
        else:
            E1, E2 = 10e3, 1e6    # PromptX gamma band
            use_X = False
    
        if t_common is None:
            t_common = np.geomspace(1e-3, 1e6, 800)
    
        # driver LC cube (per patch, source convention)
        L_src = self.L_X if use_X else self.L_gamma    # shape (nθ,nφ,nt)
    
        # choose observer grids per mode
        if mode == 'promptx':
            # PromptX keeps the *source* energy grid for spectra; times are geometric (no (1+z) here)
            E_plot = self.E.astype(float)          # no energy shift
            t_out  = t_common                      # no cosmological dilation
        elif mode == 'energy':
            # Physics mode: define a fixed observer energy grid = source grid / (1+z)
            E_plot = self.E.astype(float) / (1.0 + z)
            t_out  = t_common * (1.0 + z)          # observed times include (1+z)
        else:
            raise ValueError("mode must be 'promptx' or 'energy'.")
    
        nE = E_plot.size
        N_sum = np.zeros((nE, t_out.size), dtype=float)
    
        n_th, n_ph = self.theta.shape
        for i in range(n_th):
            for j in range(n_ph):
                if not np.isfinite(self.dOmega[i, j]) or self.dOmega[i, j] <= 0.0:
                    continue
    
                D = float(self.R_D[i, j])
    
                # per-patch observer times and LC amplitude
                if mode == 'promptx':
                    tij_obs = self.t_obs[i, j, :]    # (no (1+z))
                    Lij_obs = (D**4) * L_src[i, j, :]
                else:  # energy mode
                    tij_obs = self.t_obs[i, j, :] * (1.0 + z)
                    Lij_obs = (D**4) * L_src[i, j, :]
    
                m_t = np.isfinite(tij_obs) & np.isfinite(Lij_obs) & (tij_obs > 0)
                if m_t.sum() < 2:
                    continue
                L_interp = np.interp(t_out, tij_obs[m_t], Lij_obs[m_t], left=0.0, right=0.0)  # (nt,)
    
                # ----- per-patch spectral *shape* φ(E) -----
                EN_src = self.EN_E[i, j, :]  # on source grid self.E
    
                if mode == 'promptx':
                    # NO energy shift; EN_obs(E) = D^3 * EN_src(E)
                    EN_obs = (D**3) * EN_src
                    # PromptX νFν-in-linear-E normalizer (uses int_spec which multiplies by E internally)
                    S_band = int_spec(self.E, EN_obs, E_min=E1, E_max=E2)  # eV
                    if not np.isfinite(S_band) or S_band <= 0.0:
                        continue
                    phi = EN_obs / S_band  # units ~ 1/eV
                    E_for_phi = self.E     # energy variable for φ and division by E
    
                else:  # energy mode (physics/cross-band)
                    # Proper energy shift: E_src = E_obs*(1+z)/D
                    E_src_at_Eobs = E_plot * (1.0 + z) / D
                    EN_obs = (D**3) * np.interp(E_src_at_Eobs, self.E, EN_src, left=0.0, right=0.0)
                    # energy-conserving band normalizer in OBSERVER frame
                    S_band = _int_energy_1d(E_plot, EN_obs, E1, E2)  # eV
                    if not np.isfinite(S_band) or S_band <= 0.0:
                        continue
                    phi = EN_obs / S_band
                    E_for_phi = E_plot
    
                # photon spectrum contribution: N_ij(E,t) = [φ(E)/E] * L_ij(t) × dΩ
                contrib = (phi / np.maximum(E_for_phi, 1e-300))[:, None] * L_interp[None, :] * self.dOmega[i, j]
                N_sum += contrib
    
        # average over solid angle and convert to iso-eq or flux
        weight = np.sum(self.dOmega)
        N_iso = (4.0 * np.pi / weight) * N_sum  # ph s^-1 eV^-1
    
        if d_l_cm is not None:
            N_iso = N_iso / (4.0 * np.pi * d_l_cm**2)  # ph s^-1 cm^-2 eV^-1
    
        return (self.E if mode=='promptx' else E_plot), t_out, N_iso

    def time_resolved_spectrum_reserved(self,
                               band='X',
                               E_band=None,
                               t_common=None,
                               d_l_cm=None,
                               use_time_dependent_slopes=True,
                               amati_a=0.41, amati_b=0.83):
        """
        Time-resolved photon spectrum N(E,t) [ph s^-1 eV^-1] in the PromptX formalism.
    
        - Spectra: EN_obs(E, t) = D^3 * EN_src(E, t)
        - Light curves: L_obs(t) = D^4 * L_src(t)
        - νFν normalization over a *linear-E* band via helper.int_spec (PromptX style).
    
        If use_time_dependent_slopes is True, use:
          alpha(t) = -0.01253 * t - 0.3616
          beta(t)  =  0.006867 * t - 2.557
        where t is the (observer) time grid t_common [s].
    
        amati_a, amati_b are used to set a *fixed* E_p per theta-row (no φ-dependence)
        via the PromptX-like Amati relation; E0(t) = E_p / (2 + alpha(t)).
        """
    
        # ---------------- band and time grids ----------------
        if E_band is not None:
            E1, E2 = E_band
            use_X = (E2 <= 10e3)
        elif band == 'X':
            E1, E2 = 0.3e3, 10e3
            use_X = True
        else:
            E1, E2 = 10e3, 1e6
            use_X = False
    
        if t_common is None:
            t_common = np.geomspace(1e-3, 1e6, 800)
    
        E = self.E.astype(float)                 # PromptX: no energy shift
        nE = E.size
        t_out = t_common
        nt = t_out.size
    
        # Driver LC cube (per patch, source convention); apply D^4 per-patch later
        L_src = self.L_X if use_X else self.L_gamma    # (nθ, nφ, nt_src)
    
        # Precompute per-theta E_p from PromptX-like Amati relation; no φ-dependence
        # (self.e_iso_grid is 1D over theta)
        # E_p in eV:
        E_p_theta = 1e5 * 10.0**(amati_a * np.log10(np.maximum(self.e_iso_grid, 1e-300) / 1e51) + amati_b)
    
        # Build time-dependent alpha(t), beta(t) on the *observer* time grid
        if use_time_dependent_slopes:
            alpha_t = -0.01253 * t_out - 0.3616
            beta_t  =  0.006867 * t_out - 2.557
        else:
            # fallback: use the original PromptX constants (roughly)
            alpha_t = np.full_like(t_out, -1.0)
            beta_t  = np.full_like(t_out, -2.3)
    
        # Accumulator over solid angle
        N_sum = np.zeros((nE, nt), dtype=float)  # photons s^-1 eV^-1
    
        n_th, n_ph = self.theta.shape
        for i in range(n_th):
            # Per-theta peak energy (fixed in time), then E0(t) follows alpha(t)
            E_p_i = float(E_p_theta[i])                           # eV
            E0_t = E_p_i / (2.0 + alpha_t)                        # (nt,)
    
            for j in range(n_ph):
                if not np.isfinite(self.dOmega[i, j]) or self.dOmega[i, j] <= 0.0:
                    continue
    
                D = float(self.R_D[i, j])                         # Doppler ratio for this patch
    
                # Interp this patch’s LC onto t_out and apply D^4
                tij = self.t_obs[i, j, :]
                Lij_src = L_src[i, j, :]
                valid = np.isfinite(tij) & np.isfinite(Lij_src) & (tij > 0)
                if valid.sum() < 2:
                    continue
                L_interp = np.interp(t_out, tij[valid], Lij_src[valid], left=0.0, right=0.0) * (D**4)
    
                # ---- build EN_obs(E, t) with time-dependent slopes (PromptX: no energy shift) ----
                # EN_src(E, t) = E * band(E, alpha(t), beta(t), E0(t))
                # EN_obs(E, t) = D^3 * EN_src(E, t)
                EN_obs_E_t = np.empty((nE, nt), dtype=float)
                for k in range(nt):
                    # band(...) returns the *shape* ~ N(E) up to a constant; multiplying by E makes EN(E)
                    EN_obs_E_t[:, k] = (D**3) * (E * band(E, alpha_t[k], beta_t[k], E0_t[k]))
    
                # ---- PromptX νFν-in-linear-E normalization in the chosen band ----
                # S_obs_nufnu(t) = ∫_{E1}^{E2} EN_obs(E, t) *E dE (because int_spec multiplies by E internally)
                # Guard zeros to avoid division problems
                S_obs = np.empty(nt, dtype=float)
                for k in range(nt):
                    S_k = int_spec(E, EN_obs_E_t[:, k], E_min=E1, E_max=E2)
                    S_obs[k] = S_k if (np.isfinite(S_k) and S_k > 0.0) else 0.0
                # Replace zeros by tiny positive to avoid division by zero
                S_obs = np.where(S_obs > 0.0, S_obs, 1e-300)
    
                # Unit νFν-normalized shape φ(E,t) so that ∫ E φ(E,t) dE = 1 in [E1,E2]
                phi_E_t = EN_obs_E_t / S_obs[None, :]   # (nE, nt), ~ 1/eV
    
                # Photon spectrum for this patch: N_ij(E,t) = [φ(E,t)/E] * L_interp(t) × dΩ
                contrib = (phi_E_t / np.maximum(E[:, None], 1e-300)) * L_interp[None, :] * self.dOmega[i, j]
                N_sum += contrib
    
        # Average over solid angle and convert to isotropic-equivalent (or flux)
        weight = np.sum(self.dOmega)
        N_iso = (4.0 * np.pi / weight) * N_sum  # ph s^-1 eV^-1
    
        if d_l_cm is not None:
            N_iso = N_iso / (4.0 * np.pi * d_l_cm**2)  # ph s^-1 cm^-2 eV^-1
    
        return E, t_out, N_iso


    def time_resolved_spectrum_full_reserved(self,
                                    band='X',              # 'X' or 'gamma' (chooses driver LC only)
                                    E_band=None,           # [E1,E2] eV; if None use PromptX defaults per 'band'
                                    t_common=None,
                                    d_l_cm=None,           # if set, return flux (divide by 4π d_L^2)
                                    z=0.0,                 # cosmological redshift used in 'energy' mode
                                    mode='promptx',        # 'promptx' | 'energy'
                                    use_time_dependent_slopes=True,
                                    amati_a=0.41, amati_b=0.83):
        """
        N(E,t) time-resolved spectrum with optional time-dependent Band slopes.
    
        mode='promptx'  -> PromptX observer formalism:
                           - NO energy shift (use source E grid)
                           - spectra × D^3, LCs × D^4
                           - νFν normalization via helper.int_spec (linear-E convention)
        mode='energy'   -> Energy-conserving formalism:
                           - Proper E shift: E_obs = D * E_src / (1+z)
                           - spectra × D^3 at shifted energies
                           - LCs × D^4
                           - band power is energy-conserving: ∫ EN_obs(E_obs,t) dE_obs
                             and ∫ E_obs N(E_obs,t) dE_obs = L_interp(t)
        Time-dependent slopes:
            alpha(t) = -0.01253 * t - 0.3616
            beta(t)  =  0.006867 * t - 2.557
        E_p (per theta) from Amati; E0(t) = E_p / (2 + alpha(t)).
        """
    
        # --------------- band and time grids ---------------
        if E_band is not None:
            E1, E2 = E_band
            use_X = (E2 <= 10e3)
        elif band == 'X':
            E1, E2 = 0.3e3, 10e3
            use_X = True
        else:
            E1, E2 = 10e3, 1e6
            use_X = False
    
        if t_common is None:
            t_common = np.geomspace(1e-3, 1e6, 800)
    
        # driver LC cube (per patch, source convention)
        L_src = self.L_X if use_X else self.L_gamma    # shape (nθ, nφ, nt_src)
    
        # energy/time grids per mode
        if mode == 'promptx':
            E_plot = self.E.astype(float)              # no E shift
            t_out  = t_common                          # no (1+z) stretch here
        elif mode == 'energy':
            # choose an observer energy grid; use (source grid)/(1+z) so that an on-axis D≈1 maps ~1:1
            E_plot = self.E.astype(float) / max(1.0 + z, 1e-30)
            t_out  = t_common * (1.0 + z)              # observed times include (1+z)
        else:
            raise ValueError("mode must be 'promptx' or 'energy'.")
    
        nE = E_plot.size
        nt = t_out.size
    
        # per-theta E_p from Amati (eV); no phi dependence to match your current pipeline
        E_p_theta = 1e5 * 10.0**(amati_a * np.log10(np.maximum(self.e_iso_grid, 1e-300) / 1e51) + amati_b)  # (nθ,)
    
        # time-dependent slopes
        if use_time_dependent_slopes:
            alpha_t = -0.01253 * t_out - 0.3616
            beta_t  =  0.006867 * t_out - 2.557
        else:
            alpha_t = np.full(nt, -1.0)
            beta_t  = np.full(nt, -2.3)
    
        # accumulator over solid angle
        N_sum = np.zeros((nE, nt), dtype=float)  # photons s^-1 eV^-1
    
        n_th, n_ph = self.theta.shape
        for i in range(n_th):
            E_p_i = float(E_p_theta[i])                          # eV
            # E0(t) in either mode depends on alpha(t)
            E0_t = E_p_i / (2.0 + np.maximum(alpha_t, -1.999))   # guard 2+alpha→0
    
            for j in range(n_ph):
                if not np.isfinite(self.dOmega[i, j]) or self.dOmega[i, j] <= 0.0:
                    continue
    
                D = float(self.R_D[i, j])
    
                # observed times and LC amplitude for this patch
                tij_src  = self.t_obs[i, j, :]                   # geometric arrival times (source frame)
                Lij_src  = L_src[i, j, :]
                if mode == 'promptx':
                    tij_obs = tij_src
                else:
                    tij_obs = tij_src * (1.0 + z)
    
                m_t = np.isfinite(tij_obs) & np.isfinite(Lij_src) & (tij_obs > 0)
                if m_t.sum() < 2:
                    continue
    
                # LC on common grid with D^4 (per solid angle)
                L_interp = np.interp(t_out, tij_obs[m_t], Lij_src[m_t], left=0.0, right=0.0) * (D**4)
    
                # --------- build EN_obs(E_plot, t) per mode with time-dependent slopes ----------
                EN_obs = np.empty((nE, nt), dtype=float)
    
                if mode == 'promptx':
                    # No energy shift; EN_obs(E, t) = D^3 * E * band(E, α(t), β(t), E0(t))
                    for k in range(nt):
                        EN_obs[:, k] = (D**3) * (self.E * band(self.E, alpha_t[k], beta_t[k], E0_t[k]))
                    # PromptX νFν normalization in chosen band (linear-E convention via int_spec)
                    S_band = np.empty(nt, dtype=float)
                    for k in range(nt):
                        S_k = int_spec(self.E, EN_obs[:, k], E_min=E1, E_max=E2)  # (helper multiplies by E internally)
                        S_band[k] = S_k if (np.isfinite(S_k) and S_k > 0.0) else 0.0
                    S_band = np.where(S_band > 0.0, S_band, 1e-300)
    
                    # shape φ(E,t) s.t. ∫ E φ dE = 1 in [E1,E2]
                    phi = EN_obs / S_band[None, :]
    
                    # photons: N(E,t) = [φ(E,t)/E] * L(t) × dΩ
                    contrib = (phi / np.maximum(self.E[:, None], 1e-300)) * L_interp[None, :] * self.dOmega[i, j]
                    N_sum += contrib
    
                else:  # mode == 'energy' (energy-conserving, with proper E shift)
                    # Map observer energies to source energies for this patch at each time:
                    # E_src = E_obs * (1+z) / D
                    E_src = E_plot * (1.0 + z) / max(D, 1e-30)     # (nE,)
    
                    for k in range(nt):
                        # EN_src(E_src, t) = E_src * band(E_src, α(t), β(t), E0(t))
                        EN_src_k = E_src * band(E_src, alpha_t[k], beta_t[k], E0_t[k])
                        # EN_obs(E_obs, t) = D^3 * EN_src(E_src, t) evaluated at the shifted grid
                        EN_obs[:, k] = (D**3) * EN_src_k
    
                    # Energy-conserving band power in OBSERVER frame: S_band(t) = ∫ EN_obs dE_obs
                    # (so that ∫ E_obs * N(E_obs,t) dE_obs = L_interp(t))
                    from .helper import _int_energy_1d
                    S_band = np.empty(nt, dtype=float)
                    for k in range(nt):
                        S_k = _int_energy_1d(E_plot, EN_obs[:, k], E1, E2)
                        S_band[k] = S_k if (np.isfinite(S_k) and S_k > 0.0) else 0.0
                    S_band = np.where(S_band > 0.0, S_band, 1e-300)
    
                    # shape φ(E_obs,t) so that ∫ E_obs φ dE_obs = 1 in [E1,E2]
                    phi = EN_obs / S_band[None, :]
    
                    # photons in observer frame: N(E_obs,t) = [φ(E_obs,t)/E_obs] * L_interp(t) × dΩ
                    contrib = (phi / np.maximum(E_plot[:, None], 1e-300)) * L_interp[None, :] * self.dOmega[i, j]
                    N_sum += contrib
    
        # average over solid angle and convert to iso-eq or flux
        weight = np.sum(self.dOmega)
        N_iso = (4.0 * np.pi / weight) * N_sum  # ph s^-1 eV^-1
    
        if d_l_cm is not None:
            N_iso = N_iso / (4.0 * np.pi * d_l_cm**2)  # ph s^-1 cm^-2 eV^-1
    
        # Return E grid appropriate for the chosen mode
        return (self.E if mode == 'promptx' else E_plot), t_out, N_iso

    def time_resolved_spectrum_spectrum_driven2(self,
                                               quantity='N',        # 'N' | 'EN' | 'nuFnu'
                                               energy_band='X',            # only affects default plotting band labels (not the calc)
                                               t_common=None,
                                               d_l_cm=None,         # if set -> flux
                                               z=0.0,
                                               mode='promptx',      # 'promptx' (no E shift) | 'energy' (with E shift)
                                               amati_a=0.41, amati_b=0.83):
        """
        Spectrum-driven N(E,t) (or EN, or νFν) with time-dependent alpha(t), beta(t).
        No φ-shape or LC normalization is used; the spectrum itself carries time-dependence.
    
        quantity:
          'N'     -> photon spectrum [ph s^-1 eV^-1] (or ph s^-1 cm^-2 eV^-1 if d_l_cm)
          'EN'    -> energy spectrum  [eV s^-1 eV^-1]  (convert to erg by × EV_TO_ERG)
          'nuFnu' -> E*EN = E^2*N per logE [erg s^-1] (or erg s^-1 cm^-2)
    
        mode:
          'promptx' : EN_obs(E) = D^3 * EN_src(E); no energy shift (mimics PromptX observer()).
          'energy'  : EN_obs(E_obs) from shifting E_src = E_obs*(1+z)/D; still include D^3; times dilate by (1+z).
    
        Returns
        -------
        E_out : (nE,)  eV
        t_out : (nt,)  s
        Z     : (nE, nt)   in units per 'quantity'
        units : str
        """
    
        # ---- alpha(t), beta(t) relations ----
        def alpha_of_t(t):  # t in s
            return -0.01253*np.asarray(t) - 0.3616
        def beta_of_t(t):
            return  0.006867*np.asarray(t) - 2.557
    
        # ---- observer time grid ----
        if t_common is None:
            t_out = np.geomspace(1e-3, 1e6, 800)
        else:
            t_out = np.asarray(t_common, dtype=float)
    
        # ---- energy grid to report on ----
        # We stick to your self.E (observer grid in 'promptx'; redshifted in 'energy' mode)
        if mode == 'promptx':
            E_out = self.E.astype(float)
        elif mode == 'energy':
            E_out = self.E.astype(float) / (1.0 + z)  # fixed observer energy grid
        else:
            raise ValueError("mode must be 'promptx' or 'energy'.")
    
        nE = E_out.size
        Z_EN_sum = np.zeros((nE, t_out.size), dtype=float)  # we'll first accumulate EN (energy spectrum)
    
        # ---- build Ep(θ) via Amati and expand over φ ----
        # self.e_iso_grid is 1D in θ (per your normalize()); map over θ rows:
        E_p_theta = 1e5 * 10**(amati_a * np.log10(self.e_iso_grid / 1e51) + amati_b)  # eV
        # repeat along φ to shape (nθ, nφ)
        E_p_grid = np.repeat(E_p_theta[:, None], self.theta.shape[1], axis=1)
    
        # ---- integrate per patch and sum onto common time grid ----
        n_th, n_ph = self.theta.shape
        for i in range(n_th):
            for j in range(n_ph):
                if not np.isfinite(self.dOmega[i, j]) or self.dOmega[i, j] <= 0.0:
                    continue
    
                # observed times for this patch (PromptX EATS already computed)
                t_obs_ij = self.t_obs[i, j, :].astype(float)
                m_t = np.isfinite(t_obs_ij) & (t_obs_ij > 0)
                if m_t.sum() < 2:
                    continue
                t_patch = t_obs_ij[m_t]
                if mode == 'energy':
                    t_patch = t_patch * (1.0 + z)
    
                # time-dependent α(t), β(t)
                a_t = alpha_of_t(t_patch)
                b_t = beta_of_t(t_patch)
    
                # keep Ep fixed per patch; adjust E0(t) via Ep = (2+α)E0
                Ep = float(E_p_grid[i, j])
                E0_t = Ep / (2.0 + a_t)  # (nt_patch,)
    
                # choose which E grid to evaluate EN on for this patch
                if mode == 'promptx':
                    # no energy shift; evaluate on source=observer grid self.E
                    E_eval = self.E.astype(float)
                    # build EN_src(E,t): shape (nE, nt_patch)
                    EN_src = np.empty((E_eval.size, t_patch.size), dtype=float)
                    for k in range(t_patch.size):
                        EN_src[:, k] = E_eval * band(E_eval, a_t[k], b_t[k], E0_t[k])
                    # Doppler boost (PromptX): multiply by D^3, NO energy shift
                    D = float(self.R_D[i, j])
                    EN_obs = (D**3) * EN_src
                    # interpolate each energy slice to common t_out and accumulate × dΩ
                    for ie in range(nE):
                        Z_EN_sum[ie] += np.interp(t_out, t_patch, EN_obs[ie, :], left=0.0, right=0.0) * self.dOmega[i, j]
    
                else:
                    # energy mode: we want EN on the observer grid E_out.
                    # Build EN_obs(E_out, t): compute EN_src at E_src = E_out*(1+z)/D, then × D^3
                    D = float(self.R_D[i, j])
                    E_src_for_Eout = E_out * (1.0 + z) / D  # (nE,)
                    EN_obs = np.empty((nE, t_patch.size), dtype=float)
                    for k in range(t_patch.size):
                        # EN_src at shifted energy using the band function
                        EN_src_k = E_src_for_Eout * band(E_src_for_Eout, a_t[k], b_t[k], E0_t[k])
                        EN_obs[:, k] = (D**3) * EN_src_k
    
                    for ie in range(nE):
                        Z_EN_sum[ie] += np.interp(t_out, t_patch, EN_obs[ie, :], left=0.0, right=0.0) * self.dOmega[i, j]
    
        # ---- average over solid angle and convert to iso-eq (or flux) ----
        weight = np.sum(self.dOmega)
        EN_iso = (4.0 * np.pi / weight) * Z_EN_sum  # [eV s^-1 eV^-1]
        # Convert to flux if distance provided
        if d_l_cm is not None:
            EN_iso = EN_iso / (4.0 * np.pi * d_l_cm**2)  # [eV s^-1 cm^-2 eV^-1]
    
        # ---- deliver the requested quantity ----
        if quantity == 'EN':
            Z = EN_iso
            units = r'eV s$^{-1}$ eV$^{-1}$' if d_l_cm is None else r'eV s$^{-1}$ cm$^{-2}$ eV$^{-1}$'
        elif quantity == 'N':
            Z = EN_iso / np.maximum(E_out[:, None], 1e-300)
            units = r'ph s$^{-1}$ eV$^{-1}$' if d_l_cm is None else r'ph s$^{-1}$ cm$^{-2}$ eV$^{-1}$'
        elif quantity == 'nuFnu':
            # νFν = E * EN, convert to erg/s (or erg/s/cm^2)
            Z = (E_out[:, None] * EN_iso) * EV_TO_ERG
            units = r'erg s$^{-1}$' if d_l_cm is None else r'erg s$^{-1}$ cm$^{-2}$'
        else:
            raise ValueError("quantity must be 'N', 'EN', or 'nuFnu'.")
    
        return E_out, t_out, Z, units

    def time_resolved_spectrum_spectrum_driven(self,
                                               quantity='N',          # 'N' | 'EN' | 'nuFnu'
                                               energy_band='X',       # label only
                                               t_common=None,
                                               d_l_cm=None,           # if set -> flux
                                               z=0.0,
                                               mode='promptx',        # 'promptx' (no E shift) | 'energy' (with E shift)
                                               amati_a=0.41, amati_b=0.83):
    
        # --- time-dependent α(t), β(t) ---
        def alpha_of_t(t):  # seconds -> dimensionless slope
            t = np.asarray(t, dtype=float)
            return -0.01253*t - 0.3616
        def beta_of_t(t):
            t = np.asarray(t, dtype=float)
            return  0.006867*t - 2.557
    
        # --- common observer time grid ---
        if t_common is None:
            t_out = np.geomspace(1e-3, 1e6, 800)
        else:
            t_out = np.asarray(t_common, dtype=float)
    
        # --- output energy grid ---
        if mode == 'promptx':
            E_out = self.E.astype(float)                     # keep source grid
        elif mode == 'energy':
            E_out = self.E.astype(float) / (1.0 + z)         # fixed observer grid
        else:
            raise ValueError("mode must be 'promptx' or 'energy'.")
    
        nE, nt = E_out.size, t_out.size
        EN_accum = np.zeros((nE, nt), dtype=float)           # accumulate EN (energy spectrum)
    
        # --- precompute Ep(θ) via Amati, expand over φ ---
        # self.e_iso_grid is (nθ,); replicate across φ to match patch layout
        E_p_theta = 1e5 * 10**(amati_a * np.log10(self.e_iso_grid / 1e51) + amati_b)  # eV
        E_p_grid  = np.repeat(E_p_theta[:, None], self.theta.shape[1], axis=1)
    
        n_th, n_ph = self.theta.shape
    
        # Optional small optimization: cache arrays once
        dOmega = self.dOmega
        R_D    = self.R_D
        t_obs  = self.t_obs
    
        for i in range(n_th):
            for j in range(n_ph):
                w = dOmega[i, j]
                if not np.isfinite(w) or w <= 0.0:
                    continue
    
                # observed times for patch (already includes EATS)
                t_patch = t_obs[i, j, :].astype(float)
                m = np.isfinite(t_patch) & (t_patch > 0)
                if m.sum() < 2:
                    continue
                t_patch = t_patch[m]
                if mode == 'energy':
                    t_patch = t_patch * (1.0 + z)
    
                # interpolate α(t), β(t) and E0(t) ONCE onto t_out
                a_t = alpha_of_t(t_patch)
                b_t = beta_of_t(t_patch)
                Ep  = float(E_p_grid[i, j])
                E0_t = Ep / (2.0 + a_t)
    
                a = np.interp(t_out, t_patch, a_t, left=a_t[0], right=a_t[-1])
                b = np.interp(t_out, t_patch, b_t, left=b_t[0], right=b_t[-1])
                E0 = np.interp(t_out, t_patch, E0_t, left=E0_t[0], right=E0_t[-1])
    
                # build energy grid to evaluate Band on (source or shifted)
                D = float(R_D[i, j])
                if mode == 'promptx':
                    E_eval = E_out[:, None]                        # (nE,1), no shift
                    EN_src = E_eval * band_broadcast(E_eval, a[None, :], b[None, :], E0[None, :])
                    EN_obs = (D**3) * EN_src                       # Doppler amp only
                else:
                    # energy mode: evaluate source at E_src = E_out*(1+z)/D
                    E_src = (E_out * (1.0 + z) / D)[:, None]       # (nE,1)
                    EN_src = E_src * band_broadcast(E_src, a[None, :], b[None, :], E0[None, :])
                    EN_obs = (D**3) * EN_src
    
                EN_accum += EN_obs * w
    
        # --- average over solid angle; convert to iso-eq (or flux) ---
        weight = np.sum(dOmega)
        EN_iso = (4.0 * np.pi / weight) * EN_accum            # [eV s^-1 eV^-1]
        if d_l_cm is not None:
            EN_iso = EN_iso / (4.0 * np.pi * d_l_cm**2)       # [eV s^-1 cm^-2 eV^-1]
    
        # --- deliver requested quantity ---
        if quantity == 'EN':
            Z = EN_iso
            units = r'eV s$^{-1}$ eV$^{-1}$' if d_l_cm is None else r'eV s$^{-1}$ cm$^{-2}$ eV$^{-1}$'
        elif quantity == 'N':
            Z = EN_iso / np.maximum(E_out[:, None], 1e-300)
            units = r'ph s$^{-1}$ eV$^{-1}$' if d_l_cm is None else r'ph s$^{-1}$ cm$^{-2}$ eV$^{-1}$'
        elif quantity == 'nuFnu':
            Z = (E_out[:, None] * EN_iso) * EV_TO_ERG
            units = r'erg s$^{-1}$' if d_l_cm is None else r'erg s$^{-1}$ cm$^{-2}$'
        else:
            raise ValueError("quantity must be 'N', 'EN', or 'nuFnu'.")
    
        return E_out, t_out, Z, units


    def time_resolved_spectrum_spectrum_driven_fix(self,
                                               quantity='N',          # 'N' | 'EN' | 'nuFnu'
                                               energy_band='X',       # label only
                                               t_common=None,
                                               d_l_cm=None,           # if set -> flux
                                               z=0.0,
                                               mode='promptx',        # 'promptx' (no E shift) | 'energy' (with E shift)
                                               amati_a=0.41, amati_b=0.83):
    
        # --- time-dependent α(t), β(t) ---
        alpha_of_t=-1
        beta_of_t=-2.3
    
        # --- common observer time grid ---
        if t_common is None:
            t_out = np.geomspace(1e-3, 1e6, 800)
        else:
            t_out = np.asarray(t_common, dtype=float)
    
        # --- output energy grid ---
        if mode == 'promptx':
            E_out = self.E.astype(float)                     # keep source grid
        elif mode == 'energy':
            E_out = self.E.astype(float) / (1.0 + z)         # fixed observer grid
        else:
            raise ValueError("mode must be 'promptx' or 'energy'.")
    
        nE, nt = E_out.size, t_out.size
        EN_accum = np.zeros((nE, nt), dtype=float)           # accumulate EN (energy spectrum)
    
        # --- precompute Ep(θ) via Amati, expand over φ ---
        # self.e_iso_grid is (nθ,); replicate across φ to match patch layout
        E_p_theta = 1e5 * 10**(amati_a * np.log10(self.e_iso_grid / 1e51) + amati_b)  # eV
        E_p_grid  = np.repeat(E_p_theta[:, None], self.theta.shape[1], axis=1)
    
        n_th, n_ph = self.theta.shape
    
        # Optional small optimization: cache arrays once
        dOmega = self.dOmega
        R_D    = self.R_D
        t_obs  = self.t_obs
    
        for i in range(n_th):
            for j in range(n_ph):
                w = dOmega[i, j]
                if not np.isfinite(w) or w <= 0.0:
                    continue
    
                # observed times for patch (already includes EATS)
                t_patch = t_obs[i, j, :].astype(float)
                m = np.isfinite(t_patch) & (t_patch > 0)
                if m.sum() < 2:
                    continue
                t_patch = t_patch[m]
                if mode == 'energy':
                    t_patch = t_patch * (1.0 + z)
    
                # interpolate α(t), β(t) and E0(t) ONCE onto t_out
                a_t = alpha_of_t
                b_t = beta_of_t
                Ep  = float(E_p_grid[i, j])
                E0_t = Ep / (2.0 + a_t)
    
                a = np.interp(t_out, t_patch, a_t, left=a_t, right=a_t)
                b = np.interp(t_out, t_patch, b_t, left=b_t, right=b_t)
                E0 = np.interp(t_out, t_patch, E0_t, left=E0_t[0], right=E0_t[-1])
    
                # build energy grid to evaluate Band on (source or shifted)
                D = float(R_D[i, j])
                if mode == 'promptx':
                    E_eval = E_out[:, None]                        # (nE,1), no shift
                    EN_src = E_eval * band_broadcast(E_eval, a[None, :], b[None, :], E0[None, :])
                    EN_obs = (D**3) * EN_src                       # Doppler amp only
                else:
                    # energy mode: evaluate source at E_src = E_out*(1+z)/D
                    E_src = (E_out * (1.0 + z) / D)[:, None]       # (nE,1)
                    EN_src = E_src * band_broadcast(E_src, a[None, :], b[None, :], E0[None, :])
                    EN_obs = (D**3) * EN_src
    
                EN_accum += EN_obs * w
    
        # --- average over solid angle; convert to iso-eq (or flux) ---
        weight = np.sum(dOmega)
        EN_iso = (4.0 * np.pi / weight) * EN_accum            # [eV s^-1 eV^-1]
        if d_l_cm is not None:
            EN_iso = EN_iso / (4.0 * np.pi * d_l_cm**2)       # [eV s^-1 cm^-2 eV^-1]
    
        # --- deliver requested quantity ---
        if quantity == 'EN':
            Z = EN_iso
            units = r'eV s$^{-1}$ eV$^{-1}$' if d_l_cm is None else r'eV s$^{-1}$ cm$^{-2}$ eV$^{-1}$'
        elif quantity == 'N':
            Z = EN_iso / np.maximum(E_out[:, None], 1e-300)
            units = r'ph s$^{-1}$ eV$^{-1}$' if d_l_cm is None else r'ph s$^{-1}$ cm$^{-2}$ eV$^{-1}$'
        elif quantity == 'nuFnu':
            Z = (E_out[:, None] * EN_iso) * EV_TO_ERG
            units = r'erg s$^{-1}$' if d_l_cm is None else r'erg s$^{-1}$ cm$^{-2}$'
        else:
            raise ValueError("quantity must be 'N', 'EN', or 'nuFnu'.")
    
        return E_out, t_out, Z, units


    def time_resolved_spectrum_spectrum_driven_vector(self,
                                               quantity='N',          # 'N' | 'EN' | 'nuFnu'
                                               energy_band='X',       # label only
                                               t_common=None,
                                               d_l_cm=None,           # if set -> flux
                                               z=0.0,
                                               mode='promptx',        # 'promptx' (no E shift) | 'energy' (with E shift)
                                               amati_a=0.41, amati_b=0.83,
                                               patch_chunk=4096       # tune for memory/speed
                                               ):
    
        # --- time-dependent α(t), β(t) ---
        def alpha_of_t(t):
            t = np.asarray(t, dtype=float)
            return -0.01253*t - 0.3616
        def beta_of_t(t):
            t = np.asarray(t, dtype=float)
            return  0.006867*t - 2.557
    
        # --- common observer time grid ---
        if t_common is None:
            t_out = np.geomspace(1e-3, 1e6, 800)
        else:
            t_out = np.asarray(t_common, dtype=float)
    
        # --- output energy grid ---
        if mode == 'promptx':
            E_out = self.E.astype(float)                  # observer grid = source grid
        elif mode == 'energy':
            E_out = self.E.astype(float) / (1.0 + z)      # fixed observer grid
        else:
            raise ValueError("mode must be 'promptx' or 'energy'.")
    
        nE, nt = E_out.size, t_out.size
    
        # --- precompute α(t), β(t) once; E0 depends on Ep per patch but factorizes in time ---
        a_t = alpha_of_t(t_out)          # (nt,)
        b_t = beta_of_t(t_out)           # (nt,)
        inv_two_plus_a = 1.0/(2.0 + a_t) # (nt,)
    
        # --- Ep(θ) via Amati, broadcast over φ; flatten patches for chunking ---
        E_p_theta = 1e5 * 10**(amati_a * np.log10(self.e_iso_grid / 1e51) + amati_b)  # (nθ,)
        E_p_grid  = np.repeat(E_p_theta[:, None], self.theta.shape[1], axis=1)        # (nθ,nφ)
    
        dOmega = np.asarray(self.dOmega, dtype=float)
        R_D    = np.asarray(self.R_D,    dtype=float)
    
        # keep only useful patches
        m_patch = np.isfinite(dOmega) & (dOmega > 0.0) & np.isfinite(R_D) & (R_D > 0.0)
        if not np.any(m_patch):
            raise RuntimeError("No valid patches found.")
    
        dOmega_flat = dOmega[m_patch].ravel()                    # (nP,)
        D_flat      = R_D[m_patch].ravel()                       # (nP,)
        Ep_flat     = E_p_grid[m_patch].ravel()                  # (nP,)
        nP          = dOmega_flat.size
    
        # weights for iso-eq averaging
        weight_tot  = np.sum(dOmega_flat)
        iso_prefac  = 4.0*np.pi/weight_tot
    
        # accumulator in EN units  [eV s^-1 eV^-1]
        EN_accum = np.zeros((nE, nt), dtype=float)
    
        # --- helper: Band evaluated on array-shaped energies ---
        # Given X (any shape), a,b scalars, E0 array shaped like X[...,0] or X[...,x],
        # compute Band(X; a,b,E0) using the usual broken formula.
        def band_array(X, a, b, E0):
            # X: (..., nE) or (nP, nE); E0: (..., 1) or (...,) broadcastable to X
            xb = (a - b) * E0
            cond  = (X <= xb[..., None])  # broadcast on trailing energy axis
            low   = (X**a) * np.exp(-X / E0[..., None])
            high  = ((a - b)*E0)[..., None]**(a - b) * np.exp(b - a) * (X**b)
            return np.where(cond, low, high)
    
        # --- loop over time, vectorize over a patch CHUNK and energies ---
        # (much faster and memory-safe vs. triple loops)
        for k in range(nt):
            a = float(a_t[k]); b = float(b_t[k])
            inv = float(inv_two_plus_a[k])
    
            # process patches in chunks to keep memory bounded
            for s in range(0, nP, patch_chunk):
                e = min(s + patch_chunk, nP)
                # slice this chunk
                dOm  = dOmega_flat[s:e]                      # (nPc,)
                D3   = D_flat[s:e]**3                        # (nPc,)
                Ep   = Ep_flat[s:e]                          # (nPc,)
                E0   = (Ep * inv)                            # (nPc,)
    
                if mode == 'promptx':
                    # energies do not shift with patch
                    X = np.broadcast_to(E_out, (e - s, nE))  # (nPc, nE)
                else:
                    # energy mode: E_src = E_out*(1+z)/D
                    X = (E_out[None, :] * (1.0 + z) / D_flat[s:e][:, None])  # (nPc, nE)
    
                # EN_obs for this chunk and this time:
                # EN = D^3 * X * Band(X; a,b,E0)
                EN_chunk = (D3[:, None]) * X * band_array(X, a, b, E0)       # (nPc, nE)
    
                # sum over patches immediately and weight by dΩ
                EN_accum[:, k] += (dOm[:, None] * EN_chunk).sum(axis=0)
    
        # iso-eq (and optionally divide by 4π d_L^2 to get flux)
        EN_iso = iso_prefac * EN_accum
        if d_l_cm is not None:
            EN_iso = EN_iso / (4.0*np.pi*d_l_cm**2)
    
        # deliver requested quantity
        if quantity == 'EN':
            Z     = EN_iso
            units = r'eV s$^{-1}$ eV$^{-1}$' if d_l_cm is None else r'eV s$^{-1}$ cm$^{-2}$ eV$^{-1}$'
        elif quantity == 'N':
            Z     = EN_iso / np.maximum(E_out[:, None], 1e-300)
            units = r'ph s$^{-1}$ eV$^{-1}$' if d_l_cm is None else r'ph s$^{-1}$ cm$^{-2}$ eV$^{-1}$'
        elif quantity == 'nuFnu':
            Z     = (E_out[:, None] * EN_iso) * EV_TO_ERG
            units = r'erg s$^{-1}$' if d_l_cm is None else r'erg s$^{-1}$ cm$^{-2}$'
        else:
            raise ValueError("quantity must be 'N', 'EN', or 'nuFnu'.")
    
        return E_out, t_out, Z, units

    def time_resolved_spectrum_spectrum_driven_vector2(self,
                                               quantity='N',          # 'N' | 'EN' | 'nuFnu'
                                               energy_band='X',       # label only
                                               t_common=None,
                                               d_l_cm=None,           # if set -> flux
                                               z=0.0,
                                               mode='promptx',        # 'promptx' (no E shift) | 'energy' (with E shift)
                                               amati_a=0.41, amati_b=0.83):
    
        """
        Spectrum-driven N(E,t) with **fixed** alpha, beta.
    
        Here the spectrum has no intrinsic time dependence; all time dependence
        in Z(E,t) will be trivial (constant in t). This is essentially a
        time-independent spectrum broadcast over a chosen time grid.
    
        quantity:
          'N'     -> photon spectrum [ph s^-1 eV^-1] (or ph s^-1 cm^-2 eV^-1)
          'EN'    -> energy spectrum  [eV s^-1 eV^-1]
          'nuFnu' -> E*EN = E^2*N per logE [erg s^-1] (or erg s^-1 cm^-2)
        """
    
        # ---- fixed alpha, beta (same as in obs_grid) ----
        alpha0 = -1.0
        beta0  = -2.3
    
        # ---- time grid (just for axis; spectrum is time-independent) ----
        if t_common is None:
            t_out = np.geomspace(1e-3, 1e6, 800)
        else:
            t_out = np.asarray(t_common, dtype=float)
    
        # ---- output energy grid ----
        if mode == 'promptx':
            E_out = self.E.astype(float)                 # keep source grid
        elif mode == 'energy':
            E_out = self.E.astype(float) / (1.0 + z)     # fixed observer grid
        else:
            raise ValueError("mode must be 'promptx' or 'energy'.")
    
        nE = E_out.size
    
        # accumulate EN(E) over solid angle
        EN_accum_E = np.zeros(nE, dtype=float)
    
        # ---- Ep(θ) via Amati, expanded over φ ----
        E_p_theta = 1e5 * 10**(amati_a * np.log10(self.e_iso_grid / 1e51) + amati_b)  # eV
        E_p_grid  = np.repeat(E_p_theta[:, None], self.theta.shape[1], axis=1)
    
        dOmega = self.dOmega
        R_D    = self.R_D
    
        n_th, n_ph = self.theta.shape
        for i in range(n_th):
            for j in range(n_ph):
                w = dOmega[i, j]
                if not np.isfinite(w) or w <= 0.0:
                    continue
    
                D  = float(R_D[i, j])
                eps_ij = float(self.eps[i, j])
                Ep = float(E_p_grid[i, j])
                E0 = Ep / (2.0 + alpha0)
    
                # choose source energy grid for this patch
                if mode == 'promptx':
                    E_src = E_out                      # no energy shift
                else:  # 'energy' mode
                    E_src = E_out * (1.0 + z) / D      # E_src = E_obs(1+z)/D
    
                # EN_src(E_src) = E_src * Band(E_src)
                EN_src = E_src * band(E_src, alpha0, beta0, E0) #old
                eps_unit = int_spec(E_src, EN_src, E_min=10e3, E_max=1e6)
                if not np.isfinite(eps_unit) or eps_unit <= 0.0:
                    continue  # this patch contributes nothing in the band; skip it
                A_spec = eps_ij / eps_unit
                EN_E_norm = A_spec[..., np.newaxis] * EN_src
                # Doppler amplification
                EN_obs = (D**3) * EN_E_norm
    
                EN_accum_E += EN_obs * w
    
        # ---- average over solid angle & convert to iso-eq (or flux) ----
        weight = np.sum(dOmega)
        EN_E = (4.0 * np.pi / weight) * EN_accum_E      # [eV s^-1 eV^-1]
    
        if d_l_cm is not None:
            EN_E = EN_E / (4.0 * np.pi * d_l_cm**2)     # [eV s^-1 cm^-2 eV^-1]
    
        # broadcast in time (no intrinsic time dependence)
        EN_iso = EN_E[:, None] * np.ones_like(t_out)[None, :]
    
        # ---- return requested quantity ----
        if quantity == 'EN':
            Z = EN_iso
            units = r'eV s$^{-1}$ eV$^{-1}$' if d_l_cm is None else r'eV s$^{-1}$ cm$^{-2}$ eV$^{-1}$'
        elif quantity == 'N':
            Z = EN_iso / np.maximum(E_out[:, None], 1e-300)
            units = r'ph s$^{-1}$ eV$^{-1}$' if d_l_cm is None else r'ph s$^{-1}$ cm$^{-2}$ eV$^{-1}$'
        elif quantity == 'nuFnu':
            Z = (E_out[:, None] * EN_iso) * EV_TO_ERG
            units = r'erg s$^{-1}$' if d_l_cm is None else r'erg s$^{-1}$ cm$^{-2}$'
        else:
            raise ValueError("quantity must be 'N', 'EN', or 'nuFnu'.")
    
        return E_out, t_out, Z, units


    def time_resolved_spectrum_spectrum_driven_vector3(
            self,
            quantity='N',              # 'N' | 'EN' | 'nuFnu'
            energy_band='X',           # label only
            t_common=None,
            d_l_cm=None,               # if set -> flux
            z=0.0,
            mode='promptx',            # 'promptx' (no E shift) | 'energy' (with E shift)
            amati_a=0.41, amati_b=0.83
        ):
        """
        Spectrum-driven N(E,t) with time-dependent alpha(t), beta(t), and
        per-patch/per-time normalization to epsilon_ij over [10 keV, 1 MeV].
    
        - Spectra: EN_obs(E,t) = D^3 * EN_src(E_src,t) evaluated on the observer grid.
        - No LC normalization; time dependence comes only from alpha(t), beta(t).
        - For each patch (θ_i,φ_j) and time t_k, we scale EN_obs so that
            ∫_{10 keV}^{1 MeV} EN_obs(E,t_k) dE = eps[ij].
        """
    
        # ---- time-dependent α(t), β(t) (same relations you used) ----
        def alpha_of_t(t):  # t in s
            t = np.asarray(t, dtype=float)
            return -0.01253*t - 0.3616
        def beta_of_t(t):
            t = np.asarray(t, dtype=float)
            return  0.006867*t - 2.557
    
        # ---- observer time grid ----
        if t_common is None:
            t_out = np.geomspace(1e-3, 1e6, 800)
        else:
            t_out = np.asarray(t_common, dtype=float)
    
        # ---- output energy grid (observer frame to report on) ----
        if mode == 'promptx':
            E_out = self.E.astype(float)                 # no energy shift
        elif mode == 'energy':
            E_out = self.E.astype(float) / (1.0 + z)     # fixed observer grid
        else:
            raise ValueError("mode must be 'promptx' or 'energy'.")
    
        nE, nt = E_out.size, t_out.size
    
        # ---- Amati Ep(θ) and time-varying E0(t) = Ep / (2 + α(t)) ----
        E_p_theta = 1e5 * 10.0**(amati_a * np.log10(np.maximum(self.e_iso_grid, 1e-300)/1e51) + amati_b)  # (nθ,)
        E_p_grid  = np.repeat(E_p_theta[:, None], self.theta.shape[1], axis=1)                            # (nθ,nφ)
    
        a_t  = alpha_of_t(t_out)             # (nt,)
        b_t  = beta_of_t(t_out)              # (nt,)
        inv2 = 1.0 / (2.0 + a_t)             # (nt,)
    
        # ---- flatten valid patches for chunked vectorization ----
        dOmega = np.asarray(self.dOmega, dtype=float)
        R_D    = np.asarray(self.R_D,    dtype=float)
        eps    = np.asarray(self.eps,    dtype=float)
    
        m_patch = np.isfinite(dOmega) & (dOmega > 0.0) & np.isfinite(R_D) & (R_D > 0.0)
        if not np.any(m_patch):
            raise RuntimeError("No valid (θ,φ) patches after masking.")
    
        dOmega_flat = dOmega[m_patch].ravel()                # (nP,)
        D_flat      = R_D[m_patch].ravel()                   # (nP,)
        Ep_flat     = E_p_grid[m_patch].ravel()              # (nP,)
        eps_flat    = eps[m_patch].ravel()                   # (nP,)
        nP          = dOmega_flat.size
    
        weight_tot  = np.sum(dOmega_flat)
        iso_prefac  = 4.0*np.pi/weight_tot
    
        # ---- accumulator in EN units [eV s^-1 eV^-1] (iso-eq after averaging) ----
        EN_accum = np.zeros((nE, nt), dtype=float)
    
        # ---- vectorized Band on array-shaped energies ----
        def band_array(X, a, b, E0):
            # X: (nPc, nE), a,b scalars, E0: (nPc,)
            xb   = (a - b) * E0
            cond = X <= xb[:, None]
            low  = (X**a) * np.exp(-X / E0[:, None])
            high = (xb[:, None]**(a - b)) * np.exp(b - a) * (X**b)
            return np.where(cond, low, high)
    
        # normalization band (match obs_grid)
        E1_band, E2_band = 10e3, 1e6
        band_mask = (E_out >= E1_band) & (E_out <= E2_band)
        if not np.any(band_mask):
            # If your E grid doesn’t cover the band, nothing can be normalized
            raise ValueError("E_out does not cover 10 keV–1 MeV; expand self.E energy range.")
    
        # ---- chunked pass over patches, loop over time ----
        patch_chunk = 4096
        for k in range(nt):
            a = float(a_t[k]); b = float(b_t[k])
            inv = float(inv2[k])
    
            for s in range(0, nP, patch_chunk):
                e = min(s + patch_chunk, nP)
    
                dOm = dOmega_flat[s:e]                  # (nPc,)
                D   = D_flat[s:e]                       # (nPc,)
                D3  = D**3
                Ep  = Ep_flat[s:e]                      # (nPc,)
                E0  = Ep * inv                          # (nPc,)
                epsc = eps_flat[s:e]                    # (nPc,)
    
                # energies per patch (X is *source* energy if mode='energy', else observer energy)
                if mode == 'promptx':
                    X = np.broadcast_to(E_out, (e - s, nE))             # (nPc, nE)
                else:
                    X = (E_out[None, :] * (1.0 + z) / D[:, None])       # (nPc, nE), E_src(E_out)
    
                # unnormalized EN_obs for this time slice (per patch row)
                EN_chunk = (D3[:, None]) * X * band_array(X, a, b, E0)  # (nPc, nE)
    
                # ---- per-patch, per-time normalization to eps_ij over [10 keV, 1 MeV] (observer frame) ----
                denom = np.trapezoid(E_out[band_mask]*EN_chunk[:, band_mask], E_out[band_mask], axis=1)  # (nPc,)
                A = np.where((denom > 0) & np.isfinite(denom), epsc / denom, 0.0)       # (nPc,)
                EN_chunk *= A[:, None]
    
                # accumulate with solid-angle weights
                EN_accum[:, k] += (dOm[:, None] * EN_chunk).sum(axis=0)
    
        # ---- iso-eq average & optional flux conversion ----
        EN_iso = iso_prefac * EN_accum
        if d_l_cm is not None:
            EN_iso = EN_iso / (4.0*np.pi*d_l_cm**2)
    
        # ---- deliver requested quantity ----
        if quantity == 'EN':
            Z     = EN_iso
            units = r'eV s$^{-1}$ eV$^{-1}$' if d_l_cm is None else r'eV s$^{-1}$ cm$^{-2}$ eV$^{-1}$'
        elif quantity == 'N':
            Z     = EN_iso / np.maximum(E_out[:, None], 1e-300)
            units = r'ph s$^{-1}$ eV$^{-1}$' if d_l_cm is None else r'ph s$^{-1}$ cm$^{-2}$ eV$^{-1}$'
        elif quantity == 'nuFnu':
            Z     = (E_out[:, None] * EN_iso) * EV_TO_ERG
            units = r'erg s$^{-1}$' if d_l_cm is None else r'erg s$^{-1}$ cm$^{-2}$'
        else:
            raise ValueError("quantity must be 'N', 'EN', or 'nuFnu'.")
    
        return E_out, t_out, Z, units


    def time_resolved_spectrum_spectrum_driven_vector4(
            self,
            quantity='N',              # 'N' | 'EN' | 'nuFnu'
            energy_band='X',           # label only
            t_common=None,
            d_l_cm=None,               # if set -> flux
            z=0.0,
            mode='promptx',            # 'promptx' (no E shift) | 'energy' (with E shift)
            amati_a=0.41, amati_b=0.83
        ):
        """
        Spectrum-driven N(E,t) with time-dependent alpha(t), beta(t) and a **constant
        per-patch normalization A_ij** (Option B).
    
        - One A_ij is computed using the reference Band slopes (alpha0=-1, beta0=-2.3)
          so that ∫_{10 keV}^{1 MeV} EN_obs,ref(E) dE = eps_ij for each (theta_i,phi_j).
        - For all times, EN_obs(E,t) = A_ij * [ D^3 * E_src * Band(E_src; alpha(t),beta(t),E0(t)) ],
          evaluated on the observer energy grid (with/without energy shift per `mode`).
        - No light-curve weighting here; time dependence is purely via α(t), β(t).
    
        Returns E_out [eV], t_out [s], Z(E,t) in the requested quantity and its units.
        """
    
        # ---- time-dependent α(t), β(t) (your relations) ----
        def alpha_of_t(t):
            t = np.asarray(t, dtype=float)
            return -0.01253*t - 0.3616
        def beta_of_t(t):
            t = np.asarray(t, dtype=float)
            return  0.006867*t - 2.557
    
        # ---- observer time grid ----
        if t_common is None:
            t_out = np.geomspace(1e-3, 1e6, 800)
        else:
            t_out = np.asarray(t_common, dtype=float)
    
        # ---- output energy grid (observer frame to report on) ----
        if mode == 'promptx':
            E_out = self.E.astype(float)                 # no energy shift in E grid
        elif mode == 'energy':
            E_out = self.E.astype(float) / (1.0 + z)     # fixed observer grid
        else:
            raise ValueError("mode must be 'promptx' or 'energy'.")
    
        nE, nt = E_out.size, t_out.size
    
        # ---- Amati Ep(θ) and time-varying E0(t) = Ep / (2 + α(t)) ----
        E_p_theta = 1e5 * 10.0**(amati_a * np.log10(np.maximum(self.e_iso_grid, 1e-300)/1e51) + amati_b)  # (nθ,)
        E_p_grid  = np.repeat(E_p_theta[:, None], self.theta.shape[1], axis=1)                            # (nθ,nφ)
    
        a_t  = alpha_of_t(t_out)             # (nt,)
        b_t  = beta_of_t(t_out)              # (nt,)
        inv2 = 1.0 / (2.0 + a_t)             # (nt,)
    
        # ---- flatten valid patches for vectorization ----
        dOmega = np.asarray(self.dOmega, dtype=float)
        R_D    = np.asarray(self.R_D,    dtype=float)
        eps    = np.asarray(self.eps,    dtype=float)
    
        m_patch = np.isfinite(dOmega) & (dOmega > 0.0) & np.isfinite(R_D) & (R_D > 0.0)
        if not np.any(m_patch):
            raise RuntimeError("No valid (θ,φ) patches after masking.")
    
        dOmega_flat = dOmega[m_patch].ravel()                # (nP,)
        D_flat      = R_D[m_patch].ravel()                   # (nP,)
        Ep_flat     = E_p_grid[m_patch].ravel()              # (nP,)
        eps_flat    = eps[m_patch].ravel()                   # (nP,)
        nP          = dOmega_flat.size
    
        weight_tot  = np.sum(dOmega_flat)
        iso_prefac  = 4.0*np.pi/weight_tot
    
        # ---- helper: Band on array-shaped energies ----
        def band_array(X, a, b, E0):
            # X: (nPc, nE), a,b scalars, E0: (nPc,)
            xb   = (a - b) * E0
            cond = X <= xb[:, None]
            low  = (X**a) * np.exp(-X / E0[:, None])
            high = (xb[:, None]**(a - b)) * np.exp(b - a) * (X**b)
            return np.where(cond, low, high)
    
        # ---- reference (constant) normalization A_ij per patch using α0,β0 ----
        alpha0, beta0 = -1.0, -2.3
        E0_ref = Ep_flat / (2.0 + alpha0)    # (nP,)
    
        # Build source-energy grid per patch for the reference spectrum
        if mode == 'promptx':
            X_ref = np.broadcast_to(E_out, (nP, nE))  # no energy shift
        else:
            X_ref = (E_out[None, :] * (1.0 + z) / np.maximum(D_flat[:, None], 1e-300))
    
        # Unnormalized reference EN_obs,ref (per patch row)
        EN_ref = (np.maximum(D_flat, 0.0)**3)[:, None] * X_ref * band_array(X_ref, alpha0, beta0, E0_ref)  # (nP,nE)
    
        # Integrate EN_ref over the band (use helper.int_spec by adding a singleton axis -> 3D)
        E1_band, E2_band = 10e3, 1e6
        EN_ref_3d = EN_ref[:, None, :]  # (nP,1,nE) so int_spec can integrate along last axis
        denom_ref = int_spec(E_out, EN_ref_3d, E_min=E1_band, E_max=E2_band)  # shape (nP,1)
        denom_ref = np.squeeze(denom_ref, axis=-1)  # (nP,)
    
        tiny = 1e-300
        A_flat = np.where((denom_ref > 0) & np.isfinite(denom_ref), eps_flat / denom_ref, 0.0)  # (nP,)
    
        # ---- accumulator in EN units [eV s^-1 eV^-1] (iso-eq after averaging) ----
        EN_accum = np.zeros((nE, nt), dtype=float)
    
        # ---- chunked pass over patches, loop over time with constant A ----
        patch_chunk = 4096
        for k in range(nt):
            a = float(a_t[k]); b = float(b_t[k])
            inv = float(inv2[k])
    
            for s in range(0, nP, patch_chunk):
                e = min(s + patch_chunk, nP)
    
                dOm = dOmega_flat[s:e]                  # (nPc,)
                D   = D_flat[s:e]                       # (nPc,)
                D3  = D**3
                Ep  = Ep_flat[s:e]                      # (nPc,)
                E0  = Ep * inv                          # (nPc,)
                A   = A_flat[s:e]                       # (nPc,)
    
                # energies per patch (X is *source* energy if mode='energy', else observer energy)
                if mode == 'promptx':
                    X = np.broadcast_to(E_out, (e - s, nE))             # (nPc, nE)
                else:
                    X = (E_out[None, :] * (1.0 + z) / np.maximum(D[:, None], 1e-300))
    
                # time-dependent EN_obs (per patch row), then apply constant A
                EN_chunk = (D3[:, None]) * X * band_array(X, a, b, E0)  # (nPc, nE)
                EN_chunk *= A[:, None]
    
                # accumulate with solid-angle weights
                EN_accum[:, k] += (dOm[:, None] * EN_chunk).sum(axis=0)
    
        # ---- iso-eq average & optional flux conversion ----
        EN_iso = iso_prefac * EN_accum
        if d_l_cm is not None:
            EN_iso = EN_iso / (4.0*np.pi*d_l_cm**2)
    
        # ---- deliver requested quantity ----
        if quantity == 'EN':
            Z     = EN_iso
            units = r'eV s$^{-1}$ eV$^{-1}$' if d_l_cm is None else r'eV s$^{-1}$ cm$^{-2}$ eV$^{-1}$'
        elif quantity == 'N':
            Z     = EN_iso / np.maximum(E_out[:, None], 1e-300)
            units = r'ph s$^{-1}$ eV$^{-1}$' if d_l_cm is None else r'ph s$^{-1}$ cm$^{-2}$ eV$^{-1}$'
        elif quantity == 'nuFnu':
            Z     = (E_out[:, None] * EN_iso) * EV_TO_ERG
            units = r'erg s$^{-1}$' if d_l_cm is None else r'erg s$^{-1}$ cm$^{-2}$'
        else:
            raise ValueError("quantity must be 'N', 'EN', or 'nuFnu'.")
    
        return E_out, t_out, Z, units

    def time_resolved_spectrum_spectrum_driven_vector5(
            self,
            quantity='N',              # 'N' | 'EN' | 'nuFnu'
            energy_band='X',           # label only
            t_common=None,
            d_l_cm=None,               # if set -> flux
            z=0.0,
            mode='promptx',            # 'promptx' (no E shift) | 'energy' (with E shift)
            amati_a=0.41, amati_b=0.83,
            patch_chunk=4096
        ):
        """
        Spectrum-driven N(E,t) with time-dependent alpha(t), beta(t) and
        per-patch, per-time normalization A_ij(t) so that, at every t_k,
    
            ∫_{10 keV}^{1 MeV} EN_obs(E, t_k) dE  (PromptX νFν-in-linear-E via helper.int_spec on EN)
            = eps[i,j]
    
        No light-curve weighting; time-dependence is carried only by α(t), β(t).
        """
    
        import numpy as np
    
        # --- time-dependent α(t), β(t) ---
        def alpha_of_t(t):
            t = np.asarray(t, dtype=float)
            return -0.01253*t - 0.3616
        def beta_of_t(t):
            t = np.asarray(t, dtype=float)
            return  0.006867*t - 2.557
    
        # --- observer time grid ---
        if t_common is None:
            t_out = np.geomspace(1e-3, 1e6, 800)
        else:
            t_out = np.asarray(t_common, dtype=float)
    
        # --- output energy grid (observer frame to report on) ---
        if mode == 'promptx':
            E_out = self.E.astype(float)                 # no energy shift in E grid
        elif mode == 'energy':
            E_out = self.E.astype(float) / (1.0 + z)     # fixed observer grid
        else:
            raise ValueError("mode must be 'promptx' or 'energy'.")
    
        nE, nt = E_out.size, t_out.size
    
        # --- Amati Ep(θ) and time-varying E0(t) = Ep / (2 + α(t)) ---
        E_p_theta = 1e5 * 10.0**(amati_a * np.log10(np.maximum(self.e_iso_grid, 1e-300)/1e51) + amati_b)  # (nθ,)
        E_p_grid  = np.repeat(E_p_theta[:, None], self.theta.shape[1], axis=1)                            # (nθ,nφ)
    
        a_t  = alpha_of_t(t_out)             # (nt,)
        b_t  = beta_of_t(t_out)              # (nt,)
        inv2 = 1.0 / (2.0 + a_t)             # (nt,)
    
        # --- flatten valid patches for chunked vectorization ---
        dOmega = np.asarray(self.dOmega, dtype=float)
        R_D    = np.asarray(self.R_D,    dtype=float)
        eps    = np.asarray(self.eps,    dtype=float)
    
        m_patch = np.isfinite(dOmega) & (dOmega > 0.0) & np.isfinite(R_D) & (R_D > 0.0)
        if not np.any(m_patch):
            raise RuntimeError("No valid (θ,φ) patches after masking.")
    
        dOmega_flat = dOmega[m_patch].ravel()                # (nP,)
        D_flat      = R_D[m_patch].ravel()                   # (nP,)
        Ep_flat     = E_p_grid[m_patch].ravel()              # (nP,)
        eps_flat    = eps[m_patch].ravel()                   # (nP,)
        nP          = dOmega_flat.size
    
        weight_tot  = np.sum(dOmega_flat)
        iso_prefac  = 4.0*np.pi/weight_tot
    
        # --- vectorized Band on array-shaped energies ---
        def band_array(X, a, b, E0):
            # X: (nPc, nE), a,b scalars, E0: (nPc,)
            xb   = (a - b) * E0
            cond = X <= xb[:, None]
            low  = (X**a) * np.exp(-X / E0[:, None])
            high = (xb[:, None]**(a - b)) * np.exp(b - a) * (X**b)
            return np.where(cond, low, high)
    
        # --- normalization band and full-range bounds ---
        E1_band, E2_band = 10e3, 1e6
        E1_full, E2_full = float(E_out.min()), float(E_out.max())
        if not (E1_full <= E1_band and E2_full >= E2_band):
            raise ValueError("E_out must cover 10 keV–1 MeV.")
        
        den_band  = np.zeros((nP, nt), dtype=float)
        den_full  = np.zeros((nP, nt), dtype=float)
        
        for k in range(nt):
            a = float(a_t[k]); b = float(b_t[k]); inv = float(inv2[k])
        
            for s in range(0, nP, patch_chunk):
                e = min(s + patch_chunk, nP)
        
                D   = D_flat[s:e]                     # (nPc,)
                D3  = D**3
                E0  = Ep_flat[s:e] * inv              # (nPc,)
        
                # energies per patch (X is *source* energy if mode='energy', else observer energy)
                if mode == 'promptx':
                    X = np.broadcast_to(E_out, (e - s, nE))                  # (nPc, nE)
                else:
                    X = (E_out[None, :] * (1.0 + z) / np.maximum(D[:, None], 1e-300))
        
                # unnormalized EN_obs for this time slice (per patch row)
                EN_unnorm = (D3[:, None]) * X * band_array(X, a, b, E0)      # (nPc, nE)
        
                # use helper.int_spec on EN (νFν-in-linear-E)
                EN_3d = EN_unnorm[:, None, :]  # (nPc,1,nE)
        
                den_b = int_spec(E_out, EN_3d, E_min=E1_band, E_max=E2_band)[:, 0]   # (nPc,)
                den_f = int_spec(E_out, EN_3d, E_min=E1_full, E_max=E2_full)[:, 0]   # (nPc,)
        
                den_band[s:e, k] = den_b
                den_full[s:e, k] = den_f
        
        # per-time, per-patch normalization with overlap test
        tiny   = 1e-300
        fmin   = 1e-3        # try 1e-3 … 1e-2 (tune)
        overlap = den_band / np.maximum(den_full, tiny)
        
        use_band = overlap >= fmin
        den_used = np.where(use_band, np.maximum(den_band, tiny), np.maximum(den_full, tiny))  # (nP, nt)
        
        A_time = np.where(np.isfinite(den_used), eps_flat[:, None] / den_used, 0.0)  # (nP, nt)

    
        # =========================
        # Pass 2: build EN_accum with A_time(t)
        # =========================
        EN_accum = np.zeros((nE, nt), dtype=float)
    
        for k in range(nt):
            a = float(a_t[k]); b = float(b_t[k]); inv = float(inv2[k])
    
            for s in range(0, nP, patch_chunk):
                e = min(s + patch_chunk, nP)
    
                dOm = dOmega_flat[s:e]                 # (nPc,)
                D   = D_flat[s:e]                      # (nPc,)
                D3  = D**3
                E0  = Ep_flat[s:e] * inv               # (nPc,)
                A   = A_time[s:e, k]                   # (nPc,)
    
                if mode == 'promptx':
                    X = np.broadcast_to(E_out, (e - s, nE))                    # (nPc, nE)
                else:
                    X = (E_out[None, :] * (1.0 + z) / np.maximum(D[:, None], 1e-300))
    
                EN_chunk = (D3[:, None]) * X * band_array(X, a, b, E0)         # (nPc, nE)
                EN_chunk *= A[:, None]
    
                EN_accum[:, k] += (dOm[:, None] * EN_chunk).sum(axis=0)
    
        # ---- iso-eq average & optional flux conversion ----
        EN_iso = iso_prefac * EN_accum
        if d_l_cm is not None:
            EN_iso = EN_iso / (4.0*np.pi*d_l_cm**2)
    
        # ---- deliver requested quantity ----
        if quantity == 'EN':
            Z     = EN_iso
            units = r'eV s$^{-1}$ eV$^{-1}$' if d_l_cm is None else r'eV s$^{-1}$ cm$^{-2}$ eV$^{-1}$'
        elif quantity == 'N':
            Z     = EN_iso / np.maximum(E_out[:, None], 1e-300)
            units = r'ph s$^{-1}$ eV$^{-1}$' if d_l_cm is None else r'ph s$^{-1}$ cm$^{-2}$ eV$^{-1}$'
        elif quantity == 'nuFnu':
            Z     = (E_out[:, None] * EN_iso) * EV_TO_ERG
            units = r'erg s$^{-1}$' if d_l_cm is None else r'erg s$^{-1}$ cm$^{-2}$'
        else:
            raise ValueError("quantity must be 'N', 'EN', or 'nuFnu'.")
    
        return E_out, t_out, Z, units

    def time_resolved_spectrum_spectrum_driven_vector6(
            self,
            quantity='N',              # 'N' | 'EN' | 'nuFnu'
            energy_band='X',           # label only
            t_common=None,
            d_l_cm=None,               # if set -> flux
            z=0.0,
            mode='promptx',            # 'promptx' (no E shift) | 'energy' (with E shift)
            amati_a=0.41, amati_b=0.83,
            patch_chunk=4096
        ):
        """
        Spectrum-driven N(E,t) with time-dependent alpha(t), beta(t) and
        per-patch, per-time normalization A_ij(t) so that, at every t_k,
    
            ∫_{10 keV}^{1 MeV} EN_obs(E, t_k) dE  (PromptX νFν-in-linear-E via helper.int_spec on EN)
            = eps[i,j]
    
        BUT: to avoid huge A when the spectrum barely overlaps the band, we
        fall back to normalizing over the full energy grid at that time for
        that patch (so the total energy = eps[i,j], and the band gets its
        natural fraction). No light-curve weighting here.
    
        Returns
        -------
        E_out : (nE,)  eV
        t_out : (nt,)  s
        Z     : (nE, nt) in units per 'quantity'
        units : str
        """
        import numpy as np
    
        # --- time-dependent α(t), β(t) ---
        def alpha_of_t(t):
            t = np.asarray(t, dtype=float)
            return -0.01253*t - 0.3616
        def beta_of_t(t):
            t = np.asarray(t, dtype=float)
            return  0.006867*t - 2.557
    
        # --- observer time grid ---
        if t_common is None:
            t_out = np.geomspace(1e-3, 1e6, 800)
        else:
            t_out = np.asarray(t_common, dtype=float)
    
        # --- output energy grid (observer frame to report on) ---
        if mode == 'promptx':
            E_out = self.E.astype(float)                 # no energy shift in E grid
        elif mode == 'energy':
            E_out = self.E.astype(float) / (1.0 + z)     # fixed observer grid
        else:
            raise ValueError("mode must be 'promptx' or 'energy'.")
    
        nE, nt = E_out.size, t_out.size
    
        # --- Amati Ep(θ) and time-varying E0(t) = Ep / (2 + α(t)) ---
        E_p_theta = 1e5 * 10.0**(amati_a * np.log10(np.maximum(self.e_iso_grid, 1e-300)/1e51) + amati_b)  # (nθ,)
        E_p_grid  = np.repeat(E_p_theta[:, None], self.theta.shape[1], axis=1)                            # (nθ,nφ)
    
        a_t  = alpha_of_t(t_out)             # (nt,)
        b_t  = beta_of_t(t_out)              # (nt,)
        inv2 = 1.0 / (2.0 + a_t)             # (nt,)
    
        # --- flatten valid patches for chunked vectorization ---
        dOmega = np.asarray(self.dOmega, dtype=float)
        R_D    = np.asarray(self.R_D,    dtype=float)
        eps    = np.asarray(self.eps,    dtype=float)
    
        m_patch = np.isfinite(dOmega) & (dOmega > 0.0) & np.isfinite(R_D) & (R_D > 0.0)
        if not np.any(m_patch):
            raise RuntimeError("No valid (θ,φ) patches after masking.")
    
        dOmega_flat = dOmega[m_patch].ravel()                # (nP,)
        D_flat      = R_D[m_patch].ravel()                   # (nP,)
        Ep_flat     = E_p_grid[m_patch].ravel()              # (nP,)
        eps_flat    = eps[m_patch].ravel()                   # (nP,)
        nP          = dOmega_flat.size
    
        weight_tot  = np.sum(dOmega_flat)
        iso_prefac  = 4.0*np.pi/weight_tot
    
        # --- vectorized Band on array-shaped energies ---
        def band_array(X, a, b, E0):
            # X: (nPc, nE), a,b scalars, E0: (nPc,)
            xb   = (a - b) * E0
            cond = X <= xb[:, None]
            low  = (X**a) * np.exp(-X / E0[:, None])
            high = (xb[:, None]**(a - b)) * np.exp(b - a) * (X**b)
            return np.where(cond, low, high)
    
        # --- normalization band and full-range bounds ---
        E1_band, E2_band = 10e3, 1e6
        E1_full, E2_full = float(E_out.min()), float(E_out.max())
        if not (E1_full <= E1_band and E2_full >= E2_band):
            raise ValueError("E_out must cover 10 keV–1 MeV.")
    
        # =========================
        # Pass 1: compute den_band and den_full (nP, nt)
        # =========================
        den_band = np.zeros((nP, nt), dtype=float)
        den_full = np.zeros((nP, nt), dtype=float)
    
        for k in range(nt):
            a = float(a_t[k]); b = float(b_t[k]); inv = float(inv2[k])
    
            for s in range(0, nP, patch_chunk):
                e = min(s + patch_chunk, nP)
    
                D   = D_flat[s:e]                     # (nPc,)
                D3  = D**3
                E0  = Ep_flat[s:e] * inv              # (nPc,)
    
                # energies per patch (X is *source* energy if mode='energy', else observer energy)
                if mode == 'promptx':
                    X = np.broadcast_to(E_out, (e - s, nE))                  # (nPc, nE)
                else:
                    X = (E_out[None, :] * (1.0 + z) / np.maximum(D[:, None], 1e-300))
    
                # unnormalized EN_obs for this time slice (per patch row)
                EN_unnorm = (D3[:, None]) * X * band_array(X, a, b, E0)      # (nPc, nE)
    
                # helper.int_spec on EN (νFν-in-linear-E)
                EN_3d = EN_unnorm[:, None, :]  # (nPc,1,nE)
    
                den_b = int_spec(E_out, EN_3d, E_min=E1_band, E_max=E2_band)[:, 0]   # (nPc,)
                den_f = int_spec(E_out, EN_3d, E_min=E1_full, E_max=E2_full)[:, 0]   # (nPc,)
    
                den_band[s:e, k] = den_b
                den_full[s:e, k] = den_f
    
        # per-time, per-patch normalization with overlap safeguard
        tiny    = 1e-300
        fmin    = 1e-3    # minimum acceptable band overlap fraction; tune 1e-3..1e-2
        overlap = den_band / np.maximum(den_full, tiny)
    
        use_band = overlap >= fmin
        den_used = np.where(use_band,
                            np.maximum(den_band, tiny),
                            np.maximum(den_full, tiny))    # (nP, nt)
    
        A_time = np.where(np.isfinite(den_used), eps_flat[:, None] / den_used, 0.0)  # (nP, nt)
    
        # =========================
        # Pass 2: build EN_accum with A_time(t)
        # =========================
        EN_accum = np.zeros((nE, nt), dtype=float)
    
        for k in range(nt):
            a = float(a_t[k]); b = float(b_t[k]); inv = float(inv2[k])
    
            for s in range(0, nP, patch_chunk):
                e = min(s + patch_chunk, nP)
    
                dOm = dOmega_flat[s:e]                 # (nPc,)
                D   = D_flat[s:e]                      # (nPc,)
                D3  = D**3
                E0  = Ep_flat[s:e] * inv               # (nPc,)
                A   = A_time[s:e, k]                   # (nPc,)
    
                if mode == 'promptx':
                    X = np.broadcast_to(E_out, (e - s, nE))                  # (nPc, nE)
                else:
                    X = (E_out[None, :] * (1.0 + z) / np.maximum(D[:, None], 1e-300))
    
                EN_chunk = (D3[:, None]) * X * band_array(X, a, b, E0)       # (nPc, nE)
                EN_chunk *= A[:, None]
    
                EN_accum[:, k] += (dOm[:, None] * EN_chunk).sum(axis=0)
    
        # ---- iso-eq average & optional flux conversion ----
        EN_iso = iso_prefac * EN_accum
        if d_l_cm is not None:
            EN_iso = EN_iso / (4.0*np.pi*d_l_cm**2)
    
        # ---- deliver requested quantity ----
        if quantity == 'EN':
            Z     = EN_iso
            units = r'eV s$^{-1}$ eV$^{-1}$' if d_l_cm is None else r'eV s$^{-1}$ cm$^{-2}$ eV$^{-1}$'
        elif quantity == 'N':
            Z     = EN_iso / np.maximum(E_out[:, None], 1e-300)
            units = r'ph s$^{-1}$ eV$^{-1}$' if d_l_cm is None else r'ph s$^{-1}$ cm$^{-2}$ eV$^{-1}$'
        elif quantity == 'nuFnu':
            Z     = (E_out[:, None] * EN_iso) * EV_TO_ERG
            units = r'erg s$^{-1}$' if d_l_cm is None else r'erg s$^{-1}$ cm$^{-2}$'
        else:
            raise ValueError("quantity must be 'N', 'EN', or 'nuFnu'.")
    
        return E_out, t_out, Z, units


    def time_resolved_spectrum_spectrum_driven_vector7(
            self,
            quantity='N',              # 'N' | 'EN' | 'nuFnu'
            energy_band='X',           # label only
            t_common=None,
            d_l_cm=None,               # if set -> flux
            z=0.0,
            mode='promptx',            # 'promptx' (no E shift) | 'energy' (with E shift)
            amati_a=0.41, amati_b=0.83,
            patch_chunk=4096
        ):
        """
        Spectrum-driven N(E,t) with time-dependent alpha(t), beta(t) and
        per-patch, per-time normalization A_ij(t) such that, for every time t_k,
    
            ∫_{10 keV}^{1 MeV} EN_obs(E, t_k) dE = eps[i,j] * g(t_k),
    
        where g(t) is a *time weight* with ∫ g(t) dt = 1 (so total band energy per patch = eps[i,j]).
        No light-curve weighting; time dependence is only via α(t), β(t) (and the chosen g(t)).
    
        Returns E_out [eV], t_out [s], Z(E,t) and units.
        """
        import numpy as np
    
        # --- α(t), β(t) (your relations) ---
        def alpha_of_t(t):
            t = np.asarray(t, dtype=float)
            return -0.01253*t - 0.3616
        def beta_of_t(t):
            t = np.asarray(t, dtype=float)
            return  0.006867*t - 2.557
    
        # --- observer time grid ---
        if t_common is None:
            t_out = np.geomspace(1e-3, 1e6, 800)
        else:
            t_out = np.asarray(t_common, dtype=float)
        nt = t_out.size
    
        # ---- time weight g(t) with ∫ g dt = 1  (flat by default) ----
        # If you want a shaped profile: replace ones_like with a positive shape (e.g., fred(t_out, τ1, τ2))
        _shape = np.ones_like(t_out, dtype=float)
        g = _shape / np.trapezoid(_shape, t_out)   # (nt,), so ∫ g dt = 1
    
        # --- output energy grid (observer frame we report on) ---
        if mode == 'promptx':
            E_out = self.E.astype(float)                 # no energy shift in grid
        elif mode == 'energy':
            E_out = self.E.astype(float) / (1.0 + z)     # fixed observer grid
        else:
            raise ValueError("mode must be 'promptx' or 'energy'.")
        nE = E_out.size
    
        # --- Amati Ep(θ) and E0(t) = Ep / (2 + α(t)) ---
        E_p_theta = 1e5 * 10.0**(amati_a * np.log10(np.maximum(self.e_iso_grid, 1e-300)/1e51) + amati_b)  # (nθ,)
        E_p_grid  = np.repeat(E_p_theta[:, None], self.theta.shape[1], axis=1)                            # (nθ,nφ)
    
        a_t  = alpha_of_t(t_out)             # (nt,)
        b_t  = beta_of_t(t_out)              # (nt,)
        inv2 = 1.0 / (2.0 + a_t)             # (nt,)
    
        # --- flatten valid patches for chunked vectorization ---
        dOmega = np.asarray(self.dOmega, dtype=float)
        R_D    = np.asarray(self.R_D,    dtype=float)
        eps    = np.asarray(self.eps,    dtype=float)
    
        m_patch = np.isfinite(dOmega) & (dOmega > 0.0) & np.isfinite(R_D) & (R_D > 0.0)
        if not np.any(m_patch):
            raise RuntimeError("No valid (θ,φ) patches after masking.")
    
        dOmega_flat = dOmega[m_patch].ravel()                # (nP,)
        D_flat      = R_D[m_patch].ravel()                   # (nP,)
        Ep_flat     = E_p_grid[m_patch].ravel()              # (nP,)
        eps_flat    = eps[m_patch].ravel()                   # (nP,)
        nP          = dOmega_flat.size
    
        weight_tot  = np.sum(dOmega_flat)
        iso_prefac  = 4.0*np.pi/weight_tot
    
        # --- vectorized Band on array-shaped energies ---
        def band_array(X, a, b, E0):
            # X: (nPc, nE), a,b scalars, E0: (nPc,)
            xb   = (a - b) * E0
            cond = X <= xb[:, None]
            low  = (X**a) * np.exp(-X / E0[:, None])
            high = (xb[:, None]**(a - b)) * np.exp(b - a) * (X**b)
            return np.where(cond, low, high)
    
        # --- normalization band must lie within E_out ---
        E1_band, E2_band = 10e3, 1e6
        if not (E_out.min() <= E1_band and E_out.max() >= E2_band):
            raise ValueError("E_out does not cover 10 keV–1 MeV; expand self.E energy range.")
    
        tiny = 1e-300
    
        # =========================
        # Pass 1: denom_time (nP, nt) via PromptX int_spec on EN
        # =========================
        denom_time = np.zeros((nP, nt), dtype=float)
    
        for k in range(nt):
            a = float(a_t[k]); b = float(b_t[k]); inv = float(inv2[k])
    
            for s in range(0, nP, patch_chunk):
                e = min(s + patch_chunk, nP)
    
                D   = D_flat[s:e]                     # (nPc,)
                D3  = D**3
                E0  = Ep_flat[s:e] * inv              # (nPc,)
    
                # source energies X for this time slice
                if mode == 'promptx':
                    X = np.broadcast_to(E_out, (e - s, nE))                   # (nPc, nE)
                else:
                    X = (E_out[None, :] * (1.0 + z) / np.maximum(D[:, None], 1e-300))
    
                # EN_obs,unnormalized (per patch row)
                EN_unnorm = (D3[:, None]) * X * band_array(X, a, b, E0)       # (nPc, nE)
    
                # PromptX convention: int_spec on EN (multiplies by E internally)
                EN_3d = EN_unnorm[:, None, :]                                 # (nPc,1,nE)
                den   = int_spec(E_out, EN_3d, E_min=E1_band, E_max=E2_band)  # (nPc,1)
                denom_time[s:e, k] = den[:, 0]
    
        # per-time target power = eps_ij * g(t_k)
        A_time = (eps_flat[:, None] * g[None, :]) / np.maximum(denom_time, tiny)
        A_time[~np.isfinite(A_time)] = 0.0
    
        # =========================
        # Pass 2: build EN_accum with A_time(t)
        # =========================
        EN_accum = np.zeros((nE, nt), dtype=float)
    
        for k in range(nt):
            a = float(a_t[k]); b = float(b_t[k]); inv = float(inv2[k])
    
            for s in range(0, nP, patch_chunk):
                e = min(s + patch_chunk, nP)
    
                dOm = dOmega_flat[s:e]                 # (nPc,)
                D   = D_flat[s:e]                      # (nPc,)
                D3  = D**3
                E0  = Ep_flat[s:e] * inv               # (nPc,)
                A   = A_time[s:e, k]                   # (nPc,)
    
                if mode == 'promptx':
                    X = np.broadcast_to(E_out, (e - s, nE))
                else:
                    X = (E_out[None, :] * (1.0 + z) / np.maximum(D[:, None], 1e-300))
    
                EN_chunk = (D3[:, None]) * X * band_array(X, a, b, E0)        # (nPc, nE)
                EN_chunk *= A[:, None]
    
                EN_accum[:, k] += (dOm[:, None] * EN_chunk).sum(axis=0)
    
        # ---- iso-eq average & optional flux conversion ----
        EN_iso = iso_prefac * EN_accum
        if d_l_cm is not None:
            EN_iso = EN_iso / (4.0*np.pi*d_l_cm**2)
    
        # ---- deliver requested quantity ----
        if quantity == 'EN':
            Z     = EN_iso
            units = r'eV s$^{-1}$ eV$^{-1}$' if d_l_cm is None else r'eV s$^{-1}$ cm$^{-2}$ eV$^{-1}$'
        elif quantity == 'N':
            Z     = EN_iso / np.maximum(E_out[:, None], 1e-300)
            units = r'ph s$^{-1}$ eV$^{-1}$' if d_l_cm is None else r'ph s$^{-1}$ cm$^{-2}$ eV$^{-1}$'
        elif quantity == 'nuFnu':
            Z     = (E_out[:, None] * EN_iso) * EV_TO_ERG
            units = r'erg s$^{-1}$' if d_l_cm is None else r'erg s$^{-1}$ cm$^{-2}$'
        else:
            raise ValueError("quantity must be 'N', 'EN', or 'nuFnu'.")
    
        return E_out, t_out, Z, units
