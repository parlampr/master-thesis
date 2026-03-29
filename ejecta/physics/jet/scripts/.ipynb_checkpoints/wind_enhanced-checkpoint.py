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

from .helper import *
from .helper import band as band_fn
from .const import *
from .magnetar_enhanced import Magnetar
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

EV_TO_ERG = 1.602176634e-12

class Wind(Magnetar):
    """
    Represents a magnetar-powered relativistic wind, with energy and Lorentz factor
    profiles structured as a function of polar angle.
    """

    def __init__(self, n_theta=1000, n_phi=100, L=1e48, g0=50, L0=1e48, theta_cut=np.pi/2, k=0, collapse=False, wind_struct=1,P_0=1e-3, M=1e45, B_p=1e15, R=1e6, eps=1e-3, eos=None,kappa=1, M_ej=2e31,v=0.3*c):
        """
        Initializes a wind model for a magnetar-powered outflow 
        and defines its coordinate grid, and energy and Lorentz factor structures

        Parameters:
            n_theta (int): Number of polar (theta) grid points (default is 1000).
            n_phi (int): Number of azimuthal (phi) grid points (default is 100).
            g0 (float): Initial Lorentz factor on the wind axis (default is 50).
            L0 (float): Initial luminosity per solid angle (will be overridden by spindown unless code line is commented out).
            theta_cut (float): Cutoff angle for the wind structure (default is pi/2).
            collapse (bool): Flag indicating whether the magnetar undergoes collapse (default is False).
        
        Initializes the following attributes:
            engine (Magnetar): Instance of the Magnetar engine used to model the wind.
            theta_grid (ndarray): 2D grid of theta values for the wind.
            phi_grid (ndarray): 2D grid of phi values for the wind.
            theta (ndarray): 1D array of cell-centered theta values.
            phi (ndarray): 1D array of cell-centered phi values.
            dOmega (ndarray): Differential solid angle for each grid cell.
            theta_cut (float): Cutoff angle for the wind structure.
            eta (float): Conversion efficiency for the magnetar spin-down process.
            eps (ndarray): Energy per solid angle profile of the wind.
            g (ndarray): Lorentz factor profile of the wind.
        """

        # Create a magnetar engine instance

        self.engine = Magnetar(P_0=P_0, M=M, B_p=B_p, R=R, eps=eps, eos=None, kappa=kappa, M_ej=M_ej, v=v, collapse=collapse)

        # Define the bounds for theta (polar angle) and phi (azimuthal angle)
        theta_bounds = [0, np.pi]
        phi_bounds = [0, 2 * np.pi]

        # Generate a grid for theta and phi, based on the specified grid points
        self.theta_grid, self.phi_grid = coord_grid(n_theta, n_phi, theta_bounds, phi_bounds)

        # Compute cell-centered theta and phi values
        self.theta = (self.theta_grid[:-1, :-1] + self.theta_grid[1:, 1:]) / 2
        self.phi = (self.phi_grid[:-1, :-1] + self.phi_grid[1:, 1:]) / 2

        # Compute the differential solid angle (dOmega) for each grid cell
        dtheta = np.gradient(self.theta, axis=0)
        dphi = np.gradient(self.phi, axis=1)
        self.dOmega = np.sin(self.theta) * dtheta * dphi

        # Store core angle for wind structure (used in Gaussian/PL profile)
        self.theta_c = np.radians(20)

        self.k = k  # only used for PL profile

        # Store cutoff angle for wind structure
        self.theta_cut = theta_cut

        # Compute initial energy per solid angle from magnetar spin-down formula
        self.eta = 0.1   # Conversion efficiency
        self.define_structure(g0, L0, wind_struct)
        self.normalize(L0)

    def define_structure(self, g0, L0, wind_struct):
        """
        Defines the structure of the wind's energy and Lorentz factor profiles based on the specified profile type.

        Parameters:
        g0 (float): The initial Lorentz factor normalization.
        L0 (float): The initial luminosity per solid angle (before applying structure).
        E_iso (float): The isotropic-equivalent energy used for the Gaussian and power-law structures.
        struct (int or function): The structure type, where:
            - 1: Tophat
            - 2: Gaussian
            - 3: Power-law
            - function: A custom piecewise function to define eps and gamma.

        The function updates the `self.eps` and `self.g` attributes based on the selected structure.
        """

        self.g0 = g0  
        self.L0 = L0  
        self.struct = wind_struct

        if callable(self.struct):  # Custom function
            self.L = eps_grid(self.L0, self.theta, self.phi, struct=self.struct)
            self.g = gamma_grid(self.g0, self.theta, self.phi, struct=self.struct)

        elif self.struct in (1, 'tophat'):  # Tophat
            self.L = eps_grid(self.L0, self.theta, self.phi, struct='tophat', cutoff=self.theta_cut)
            self.g = gamma_grid(self.g0, self.theta, self.phi, struct='tophat', cutoff=self.theta_cut)

        elif self.struct in (2, 'gaussian'):  # Gaussian
            self.L = eps_grid(self.L0, self.theta, self.phi, theta_c=self.theta_c, struct='gaussian', cutoff=self.theta_cut)
            self.g = gamma_grid(self.g0, self.theta, self.phi, theta_c=self.theta_c, struct='gaussian', cutoff=self.theta_cut)

        elif self.struct in (3, 'powerlaw'):  # Power-law
            l = 2
            self.L = eps_grid(self.L0, self.theta, self.phi, theta_c=self.theta_c, k=l, struct='powerlaw', cutoff=self.theta_cut)
            self.g = gamma_grid(self.g0, self.theta, self.phi, theta_c=self.theta_c, k=l, struct='powerlaw', cutoff=self.theta_cut)

        else: 
            raise ValueError(f"Unsupported jet structure type: {self.struct}. Use 'tophat', 'gaussian', 'powerlaw', or a custom function.")

    def normalize(self, L0):
        """
        Normalizes the wind luminosity profile to match the given luminosity.

        This method scales the jet's luminosity profile (per solid angle) such that the observed on-axis 
        luminosity equals a given value.

        Parameters:
        L (float): The target luminosity to normalize to.
        """

        self.observer()

        # Compute the normalization factor
        A = L0 / self.L_prime_los
    
        # Apply the normalization
        self.L *= A
        
        print('Normalized L0:', self.L[0][0])


    def observer(self, theta_los=0, phi_los=0):
        """
        Compute observer-frame wind emission, including luminosity and spectra.

        Parameters:
            theta_los (float): Observer polar angle (line of sight).
            phi_los (float): Observer azimuthal angle.
        """

        # Find grid coordinates corresponding to line of sight (LoS)
        los_coord = nearest_coord(self.theta, self.phi, theta_los, phi_los)

        # Compute Doppler factors
        dopp_on = doppf(self.g, 0)  # On-axis Doppler factor
        dopp_off = doppf(
            self.g,
            angular_d(self.theta[los_coord[0], los_coord[1]], self.theta,
                    self.phi[los_coord[0], los_coord[1]], self.phi)
        )
        self.r_dopp = dopp_off / dopp_on

        # Doppler-boosted luminosity
        self.L_prime = self.L * self.r_dopp**2
        self.L_prime4 = self.L * self.r_dopp**4
        L_obs_per_sa = self.L * self.r_dopp**4

        mask_cut = (self.theta < self.theta_cut) | (self.theta > np.pi - self.theta_cut)

        # self.L_prime_los = np.sum(L_cut * dOmega_cut * r_dopp_cut**0)
        # self.L_prime_los_full = np.sum(self.L_prime * self.dOmega * self.r_dopp**0) 

        self.L_prime_los = np.sum(L_obs_per_sa[mask_cut] * self.dOmega[mask_cut])
        self.L_prime_los_full = np.sum(L_obs_per_sa * self.dOmega)

        weight = np.sum(self.dOmega )

        self.L_prime_los /= weight
        self.L_prime_los_full /= weight

        # print('Total L_eff in los:', self.L_prime_los)

        tau = np.exp(-10 * self.engine.tau)  # Smoothly transitions from 0 to 1

        self.L_obs = self.L_prime_los_full * tau * (self.engine.Omega / self.engine.Omega_0)**4

        # Total observed luminosity
        self.L_X_tot = np.where(
            self.engine.t > self.engine.t_coll,
            0,
            (1 - tau) * self.L_prime_los * (self.engine.Omega / self.engine.Omega_0)**4 +
            tau * self.L_prime_los_full * (self.engine.Omega / self.engine.Omega_0)**4
        )



    def time_resolved_spectrum(self,
                               band='X',
                               E_band=None,
                               E_grid_eV=None,
                               alpha=-1.0,
                               beta=-2.3,
                               ep_ref_keV=5.0,
                               L_ref=1e48,
                               delta=0.5,
                               d_l_cm=None):
        """
        Time-resolved photon spectrum for the WIND component (no detector folding).
    
        Parameters
        ----------
        band : 'X' or 'gamma'
            Chooses which wind light curve to use as the time-dependent amplitude.
            - 'X'   -> uses self.L_X_tot  (0.3–10 keV band power, from your wind.observer)
            - 'gamma' -> uses self.L_obs  (a broader power proxy; adjust as you like)
        E_band : (E1, E2) in eV, optional
            Energy interval used to normalize the spectral shape. If None:
            - for 'X'   -> (0.3 keV, 10 keV)
            - for 'gamma' -> (10 keV, 1 MeV)
        E_grid_eV : array, optional
            Energy grid (eV). Default: geomspace(1e2, 1e8, 1000).
        alpha, beta : float
            Band-function indices.
        ep_ref_keV : float
            Reference peak energy (keV) at L_ref.
        L_ref : float
            Reference luminosity for the Yonetoku-like scaling.
        delta : float
            Exponent in E_p ∝ L^delta (default 0.5).
        d_l_cm : float, optional
            Luminosity distance. If provided, returns flux [ph s^-1 cm^-2 eV^-1].
    
        Returns
        -------
        E : (n_E,) in eV
        t : (n_t,) in s
        N : (n_E, n_t)
            - photons s^-1 eV^-1 (isotropic-equivalent), if d_l_cm is None
            - photons s^-1 cm^-2 eV^-1 (flux), if d_l_cm is set
        """
    
        # ---------- choose amplitude LC and normalization band ----------
        if E_band is not None:
            E1, E2 = E_band
        elif band == 'X':
            E1, E2 = 0.3e3, 10e3            # 0.3–10 keV
        else:
            E1, E2 = 10e3, 1e6              # 10 keV–1 MeV
    
        # Light curve to drive the shape (energy per time in the chosen band)
        # L_X_tot is what wind.observer() already produces; L_obs is a broader proxy
        if band == 'X':
            L_band = np.asarray(self.L_X_tot, dtype=float)   # (n_t,)
        else:
            L_band = np.asarray(self.L_obs, dtype=float)     # (n_t,)
    
        t = np.asarray(self.engine.t, dtype=float)           # (n_t,)
    
        # Guard length / non-negativity
        n_t = t.size
        L_band = np.where(np.isfinite(L_band) & (L_band >= 0), L_band, 0.0)
    
        # ---------- energy grid ----------
        E = np.asarray(E_grid_eV, dtype=float) if E_grid_eV is not None else np.geomspace(1e2, 1e8, 1000)
        n_E = E.size
    
        # ---------- time-dependent peak energy (Yonetoku-like) ----------
        # E_p(t) = ep_ref * (max(L, tiny)/L_ref)^delta
        ep_ref_eV = ep_ref_keV * 1e3
        L_safe = np.maximum(L_band, 1e-300)
        E_p_t = ep_ref_eV * (L_safe / max(L_ref, 1e-300))**delta     # (n_t,)
        E0_t = E_p_t / (2.0 + alpha)                                 # (n_t,)
    
        # ---------- build EN_E(E,t) and normalize shape in the chosen band ----------
        # EN_E_t[:,k] = E * band(E; alpha,beta,E0_t[k])  (energy density per eV, in eV units)
        EN_E_t = np.empty((n_E, n_t), dtype=float)
        for k in range(n_t):
            EN_E_t[:, k] = E * band_fn(E, alpha, beta, E0_t[k])
    
        # S_band(t) = ∫_{E1}^{E2} EN_E(E,t) dE  (energy-consistent)
        m_band = (E >= E1) & (E <= E2)
        if not np.any(m_band):
            raise ValueError("E_band does not overlap the energy grid.")
        S_band_t = np.trapezoid(EN_E_t[m_band, :], E[m_band], axis=0)   # (n_t,)
        S_band_t = np.where(S_band_t > 0, S_band_t, 1.0)
    
        # Unit-energy shape φ(E,t) = EN_E / S_band(t)  (per eV), so ∫_band φ dE = 1
        phi = EN_E_t / S_band_t[None, :]
    
        # ---------- photons: N(E,t) = [φ(E,t)/E] * L_band(t) ----------
        Einv = 1.0 / E[:, None]
        N = (phi * Einv) * L_band[None, :]                      # (n_E, n_t)
    
        # ---------- flux option ----------
        if d_l_cm is not None:
            N = N / (4.0 * np.pi * d_l_cm**2)                   # ph s^-1 cm^-2 eV^-1
    
        # Clean bad numbers
        N = np.nan_to_num(N, nan=0.0, posinf=0.0, neginf=0.0)
        return E, t, N

    def time_resolved_spectrum_const(self, band='X', E_band=None, E_grid_eV=None,
                                 alpha=-1.0, beta=-2.3, Ep_keV=5.0, d_l_cm=None):
        # choose band and L(t)
        if E_band is not None: E1,E2 = E_band
        elif band=='X':        E1,E2 = 0.3e3, 10e3
        else:                  E1,E2 = 10e3, 1e6
        L_band = self.L_X_tot if band=='X' else self.L_obs
        t = self.engine.t
    
        # energy grid
        E = np.geomspace(1e2, 1e8, 1000) if E_grid_eV is None else np.asarray(E_grid_eV)
        E0 = (Ep_keV*1e3) / (2.0 + alpha)  # Band parameter
    
        # template: EN(E) = E * N_band(E; alpha,beta,E0)
        EN = E * band_fn(E, alpha, beta, E0)       # band_fn is your helper.band aliased
        m  = (E>=E1) & (E<=E2)
        S  = np.trapezoid(EN[m], E[m])             # ∫ EN dE over band
    
        phi = EN / (S if S>0 else 1.0)             # unit-energy shape (per eV)
        N   = (phi[:,None] / E[:,None]) * L_band[None,:]  # photons s^-1 eV^-1
    
        if d_l_cm is not None:
            N = N / (4*np.pi*d_l_cm**2)            # photons s^-1 cm^-2 eV^-1

        return E, t, np.nan_to_num(N)

    
    
    def plot_time_resolved_cmap(self,
                                flag=True,
                                quantity='nuFnu',   # 'N' | 'EN' | 'nuFnu'
                                band='X',
                                E_band=None,
                                E_grid_eV=None,
                                d_l_cm=None,
                                energy_unit='keV',
                                cmap='inferno',
                                log_norm=True,
                                vmin=None,
                                vmax=None,
                                title=None):
        """
        Colormap for the WIND time-resolved spectrum.
        quantity:
          'N'     -> photons s^-1 eV^-1  (or cm^-2 eV^-1 if distance given)
          'EN'    -> erg s^-1 eV^-1      (or erg s^-1 cm^-2 eV^-1)
          'nuFnu' -> erg s^-1            (or erg s^-1 cm^-2)
        """
        if flag:
            E, t, N = self.time_resolved_spectrum(band=band, E_band=E_band,
                                                  E_grid_eV=E_grid_eV,
                                                  d_l_cm=d_l_cm)
        else:
            E, t, N = self.time_resolved_spectrum_const(band=band, E_band=E_band,
                                                  E_grid_eV=E_grid_eV,
                                                  d_l_cm=d_l_cm)
        # choose Z to plot
        if quantity == 'N':
            Z = N
            units = r'ph s$^{-1}$ eV$^{-1}$' if d_l_cm is None else r'ph s$^{-1}$ cm$^{-2}$ eV$^{-1}$'
        elif quantity == 'EN':
            Z = (E[:, None] * N) * EV_TO_ERG
            units = r'erg s$^{-1}$ eV$^{-1}$' if d_l_cm is None else r'erg s$^{-1}$ cm$^{-2}$ eV$^{-1}$'
        elif quantity == 'nuFnu':
            Z = (E[:, None]**2 * N) * EV_TO_ERG
            units = r'erg s$^{-1}$' if d_l_cm is None else r'erg s$^{-1}$ cm$^{-2}$'
        else:
            raise ValueError("quantity must be 'N', 'EN', or 'nuFnu'.")
    
        # axis units
        scale = {'eV':1.0, 'keV':1e-3, 'MeV':1e-6}.get(energy_unit, 1e-3)
        E_plot = E * scale
    
        # robust edges for log grid
        def edges_from_centers(x):
            x = np.asarray(x)
            edges = np.empty(x.size + 1)
            edges[1:-1] = np.sqrt(x[:-1] * x[1:])
            edges[0]  = x[0]**2 / edges[1]
            edges[-1] = x[-1]**2 / edges[-2]
            return edges
    
        E_edges = edges_from_centers(E_plot)
        t_edges = edges_from_centers(t)
    
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
            title = {'N':'Photon spectrum (wind)',
                     'EN':'Energy density per eV (wind)',
                     'nuFnu':'$\\nu F_\\nu$ (wind)'}[quantity]
        ax.set_title(title)
        fig.tight_layout()
        return fig, ax
