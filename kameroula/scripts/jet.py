# ============================================================================= #
#   PromptX — JAX-accelerated Jet model                                        #
#   Original authors: Connery Chen, Yihan Wang, and Bing Zhang                 #
#   License: MIT                                                               #
# ============================================================================= #

import jax
import jax.numpy as jnp
import numpy as np

from .helper import *
from .helper import _int_energy_1d, _interp_1d, band_broadcast
from .const import *

EV_TO_ERG = 1.602176634e-12


# ── JIT-compiled observer core ─────────────────────────────────────────────
# Extracted from Jet.observer() so JAX compiles the ENTIRE hot path
# (Doppler factors + EATS + interp_lc) into one fused XLA program.
# First call: ~5-15s compilation. All subsequent calls: <0.1s.

@jax.jit
def _compute_observer(theta, phi, g, t, EN_E, L_gamma, L_X, dOmega, S_gamma,
                       theta_los, phi_los):
    """JIT-compiled observer computation. Called from Jet.observer()."""

    # Nearest coord (dynamic indexing — works in JIT)
    theta_i = jnp.abs(theta[:, 0] - theta_los).argmin()
    phi_i = jnp.abs(phi[0, :] - phi_los).argmin()

    # Doppler ratio
    D_on = doppf(g, 0.0)
    D_off = doppf(g, angular_d(theta[theta_i, phi_i], theta,
                                phi[theta_i, phi_i], phi))
    R_D = D_off / D_on

    # EATS
    beta = gamma2beta(g)
    theta_obs = angular_d(theta_los, theta, phi_los, phi)
    t_lab = t[jnp.newaxis, jnp.newaxis, :]
    R_em = c * t_lab * beta[..., jnp.newaxis] / (1 - beta[..., jnp.newaxis])
    t_obs = t + R_em / c * (1 - jnp.cos(theta_obs)[..., jnp.newaxis])

    # Per-patch observed quantities × dΩ
    R_D3 = R_D[..., jnp.newaxis] ** 3
    R_D4 = R_D3 * R_D[..., jnp.newaxis]
    dOm = dOmega[..., jnp.newaxis]

    EN_E_obs = EN_E * R_D3 * dOm
    L_gamma_obs = L_gamma * R_D4 * dOm
    L_X_obs = L_X * R_D4 * dOm

    # Sum spectra
    spec_tot = 4 * jnp.pi * jnp.sum(EN_E_obs, axis=(0, 1))

    # interp_lc (vmap'd — all 50,000 patches in parallel)
    t_common, L_gamma_tot = interp_lc(t_obs, L_gamma_obs)
    _, L_X_tot = interp_lc(t_obs, L_X_obs)

    # Weight and normalize
    weight = jnp.sum(dOmega)
    L_gamma_tot = L_gamma_tot / weight * 4 * jnp.pi
    L_X_tot = L_X_tot / weight * 4 * jnp.pi
    spec_tot = spec_tot / weight

    # E_iso_obs and L_iso_obs
    eps_bar_gamma = int_spec(t_common, spec_tot, E_min=10e3, E_max=1000e3)

    S_prime = S_gamma * R_D
    S_prime3 = S_gamma * R_D ** 3

    return (t_common, L_gamma_tot, L_X_tot, R_D, t_obs,
            spec_tot, eps_bar_gamma, S_prime, S_prime3,
            EN_E_obs, L_gamma_obs, L_X_obs)


class Jet:
    """Relativistic jet model — JAX-accelerated."""

    def __init__(self, n_theta=500, n_phi=100, g0=200, E_iso=1e53, eps0=1e53,
                 theta_c=jnp.pi/2, k=0, theta_cut=jnp.pi/2, jet_struct=0):

        theta_bounds = [0, float(jnp.pi)]
        phi_bounds = [0, float(2 * jnp.pi)]

        self.theta_grid, self.phi_grid = coord_grid(n_theta, n_phi,
                                                     theta_bounds, phi_bounds)

        self.theta = 0.25 * (self.theta_grid[:-1, :-1] + self.theta_grid[1:, :-1] +
                             self.theta_grid[:-1, 1:] + self.theta_grid[1:, 1:])
        self.phi = 0.25 * (self.phi_grid[:-1, :-1] + self.phi_grid[1:, :-1] +
                           self.phi_grid[:-1, 1:] + self.phi_grid[1:, 1:])

        dtheta = jnp.gradient(self.theta, axis=0)
        dphi = jnp.gradient(self.phi, axis=1)
        self.dOmega = jnp.sin(self.theta) * dtheta * dphi

        self.theta_c = theta_c
        self.theta_cut = theta_cut
        self.k = k

        self.define_structure(eps0, E_iso, jet_struct)
        self.normalize(self.E_iso)

    def define_structure(self, eps0, E_iso, jet_struct):
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
        self.eps = eps_grid(self.eps0, self.theta, self.phi,
                            theta_c=self.theta_c, k=self.k, struct=struct, cutoff=cutoff)
        E_iso_profile = eps_grid(self.E_iso, self.theta, self.phi,
                                  theta_c=self.theta_c, k=self.k, struct=struct, cutoff=cutoff)
        self.g = lg11(E_iso_profile)

    def normalize(self, E_iso):
        e_N = e_iso_grid(self.theta, self.phi, 0.0, self.g, self.eps,
                         self.theta_cut, self.dOmega)
        e_S = e_iso_grid(self.theta, self.phi, float(jnp.pi), self.g, self.eps,
                         self.theta_cut, self.dOmega)
        self.e_iso_grid = e_N + e_S

        A = E_iso / self.e_iso_grid[0]
        self.eps = self.eps * A            # JAX arrays are immutable — no *=
        self.e_iso_grid = self.e_iso_grid * A   # Linear in eps — skip recompute

    def create_obs_grid(self, amati_a=0.41, amati_b=0.83):
        # Compute Band spectrum + FRED ONCE, integrate for both bands
        (self.E, self.EN_E, self.t, self.L_gamma, self.S_gamma,
         self.L_X, self.S_X) = obs_grid_both(
            self.eps, self.e_iso_grid, amati_a=amati_a, amati_b=amati_b)

    def observer(self, theta_los=0, phi_los=0):
        """
        JIT-compiled observer computation.
        First call compiles the XLA program (~5-15s).
        All subsequent calls: <0.1s per sample.
        """
        (self.t, self.L_gamma_tot, self.L_X_tot, self.R_D, self.t_obs,
         self.spec_tot, self.eps_bar_gamma, self.S_prime, self.S_prime3,
         self.EN_E_obs, self.L_gamma_obs, self.L_X_obs
        ) = _compute_observer(
            self.theta, self.phi, self.g, self.t, self.EN_E,
            self.L_gamma, self.L_X, self.dOmega, self.S_gamma,
            jnp.float64(theta_los), jnp.float64(phi_los)
        )

        self.E_iso_obs = self.eps_bar_gamma
        self.L_iso_obs = int_lc(self.t, self.L_gamma_tot)
