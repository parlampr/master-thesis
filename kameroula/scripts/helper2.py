# ============================================================================= #
#   PromptX — JAX-accelerated version                                          #
#   Original authors: Connery Chen, Yihan Wang, and Bing Zhang                 #
#   License: MIT                                                               #
# ============================================================================= #

import jax
import jax.numpy as jnp
import numpy as np
import csv

from .const import *


# ── Elementary functions ────────────────────────────────────────────────────

def gamma2beta(gamma):
    return jnp.sqrt(1 - 1 / gamma**2)

def beta2gamma(beta):
    return 1 / jnp.sqrt(1 - beta**2)

def gaussian(x, sigma, mu=0):
    return jnp.exp(-((x - mu)**2 / (2 * sigma**2)))

def powerlaw(x, theta_c, k):
    return jnp.where(x <= theta_c, 1.0, (x / theta_c) ** (-k))

def doppf(g, theta):
    return 1 / (g * (1 - jnp.sqrt(1 - 1 / g / g) * jnp.cos(theta)))

def fred(t, tau_1, tau_2):
    return jnp.exp((2 * (tau_1 / tau_2) ** 0.5) - (tau_1 / t + t / tau_2))

def impulse(t, t_peak, width=1e-3):
    return jnp.exp(-0.5 * ((t - t_peak) / width) ** 2)

def angular_d(theta_1, theta_2, phi_1, phi_2):
    return jnp.arccos(
        jnp.clip(
            jnp.sin(theta_1) * jnp.sin(theta_2) * jnp.cos(phi_1 - phi_2) +
            jnp.cos(theta_1) * jnp.cos(theta_2),
            -1.0, 1.0
        )
    )

def spherical_to_cartesian(theta, phi):
    return jnp.sin(theta) * jnp.cos(phi), jnp.sin(theta) * jnp.sin(phi)

def lg11(eps):
    return 200 * (eps / 1e52)**0.25 + 1

def nearest_coord(theta, phi, theta_los, phi_los):
    theta_i = jnp.abs(theta[:, 0] - theta_los).argmin()
    phi_i = jnp.abs(phi[0, :] - phi_los).argmin()
    return theta_i, phi_i


# ── Band function ──────────────────────────────────────────────────────────

def band(E, alpha, beta, E_0):
    """Band function spectrum. E_0 can be scalar or array."""
    E = jnp.asarray(E)
    E_0 = jnp.asarray(E_0)

    if E_0.ndim == 0:
        cond = E <= (alpha - beta) * E_0
        lowE = (E)**alpha * jnp.exp(-E / E_0)
        highE = ((alpha - beta) * E_0)**(alpha - beta) * jnp.exp(beta - alpha) * (E)**beta
    else:
        E_b = E[jnp.newaxis, jnp.newaxis, :]
        E_0_b = E_0[..., jnp.newaxis, jnp.newaxis]
        cond = E_b <= (alpha - beta) * E_0_b
        lowE = (E_b)**alpha * jnp.exp(-E_b / E_0_b)
        highE = ((alpha - beta) * E_0_b)**(alpha - beta) * jnp.exp(beta - alpha) * (E_b)**beta
    return jnp.where(cond, lowE, highE)


def band_broadcast(E_2d, alpha_2d, beta_2d, E0_2d):
    """Vectorized Band for (nE, nt) shaped arrays."""
    E = jnp.asarray(E_2d)
    a = jnp.asarray(alpha_2d)
    b = jnp.asarray(beta_2d)
    E0 = jnp.asarray(E0_2d)
    Eb = (a - b) * E0
    cond = E <= Eb
    low = (E**a) * jnp.exp(-E / E0)
    high = (Eb**(a - b)) * jnp.exp(b - a) * (E**b)
    return jnp.where(cond, low, high)


# ── Integration (mask-multiply approach — JIT-compatible) ──────────────────

def int_spec(E, N_E, E_min=None, E_max=None):
    """
    Integrate EN(E)*E over energy band [E_min, E_max].
    Uses mask multiplication (no dynamic slicing) for JIT compatibility.
    """
    E = jnp.asarray(E)
    N_E = jnp.asarray(N_E)
    mask = jnp.ones(E.shape[0])
    if E_min is not None:
        mask = mask * (E >= E_min)
    if E_max is not None:
        mask = mask * (E <= E_max)

    if N_E.ndim == 1:
        return jnp.trapezoid(N_E * E * mask, E)
    elif N_E.ndim == 3:
        mask_b = mask[jnp.newaxis, jnp.newaxis, :]
        E_b = E[jnp.newaxis, jnp.newaxis, :]
        return jnp.trapezoid(N_E * E_b * mask_b, E, axis=-1)
    else:
        raise ValueError("N_E must be 1D or 3D.")


def int_lc(t, L):
    """Integrate light curve over time."""
    return jnp.trapezoid(L, t)


# ── interp_lc  —  vmap'd, no Python loop ──────────────────────────────────

@jax.jit
def interp_lc(t, L):
    """
    Interpolate per-patch light curves onto common time grid and sum.
    Uses jax.vmap to process all 50,000 patches in parallel — no Python loop.

    Parameters
    ----------
    t : (n_theta, n_phi, n_t) — per-patch observed time grids
    L : (n_theta, n_phi, n_t) — per-patch luminosity values

    Returns
    -------
    t_common : (1000,)
    L_total  : (1000,)
    """
    t_common = jnp.geomspace(1e-3, 1e6, 1000)

    t_flat = t.reshape(-1, t.shape[-1])   # (n_patches, n_t)
    L_flat = L.reshape(-1, L.shape[-1])

    def interp_single(t_ij, L_ij):
        valid = jnp.all(jnp.isfinite(t_ij)) & jnp.any(L_ij > 0)
        L_interp = jnp.interp(t_common, t_ij, L_ij, left=0.0, right=0.0)
        return jnp.where(valid, L_interp, jnp.zeros_like(t_common))

    all_interped = jax.vmap(interp_single)(t_flat, L_flat)  # (n_patches, 1000)
    return t_common, jnp.sum(all_interped, axis=0)


# ── e_iso_grid  —  vmap'd, no Python loop ─────────────────────────────────

@jax.jit
def e_iso_grid(theta, phi, theta_0, g, eps, theta_cut, dOmega):
    """
    Compute isotropic-equivalent energy at each theta row.
    Uses jax.vmap over observer angles — no Python loop.
    """
    D_on = doppf(g, 0.0)
    omega_sum = jnp.sum(dOmega)
    eps_masked = jnp.where(theta < theta_cut, eps, 0.0)
    dOm_masked = jnp.where(theta < theta_cut, dOmega, 0.0)

    def compute_single(obs_th):
        ang = theta_0 - angular_d(obs_th, theta, phi[0, 0], phi)
        D_off = doppf(g, ang)
        R_D = D_off / D_on
        return 4.0 * jnp.pi * jnp.sum(eps_masked * R_D**3 * dOm_masked) / omega_sum

    return jax.vmap(compute_single)(theta[:, 0])


# ── obs_grid_both  —  shared spectrum + FRED, two bands ───────────────────

@jax.jit
def obs_grid_both(eps, e_iso_grid_arr, amati_a=0.41, amati_b=0.83):
    """
    Compute spectrum + light curve for BOTH gamma and X-ray bands in one pass.
    Band spectrum and FRED are computed once, only the integration differs.

    Returns: E, EN_E_norm, t, L_gamma, S_gamma, L_X, S_X
    """
    alpha, beta_val = -1, -2.3
    e_iso_grid_arr = jnp.asarray(e_iso_grid_arr)
    E_p = 1e5 * 10**(amati_a * jnp.log10(jnp.maximum(e_iso_grid_arr, 1e-300) / 1e51) + amati_b)
    E_0 = E_p / (2 + alpha)

    E = jnp.geomspace(1e2, 1e8, 1000)

    # Band spectrum — computed ONCE
    EN_E = E * band(E, alpha, beta_val, E_0)

    # Normalize to eps — computed ONCE
    eps_unit = int_spec(E, EN_E, E_min=10e3, E_max=1e6)
    eps_unit = jnp.where(jnp.abs(eps_unit) < 1e-300, 1e-300, eps_unit)
    A_spec = eps / eps_unit
    EN_E_norm = A_spec[..., jnp.newaxis] * EN_E

    # Gamma band (10 keV - 1 MeV)
    S_gamma = int_spec(E, EN_E_norm, E_min=10e3, E_max=1000e3)
    S_gamma = jnp.nan_to_num(S_gamma, nan=0.0)

    # X-ray band (0.3 keV - 10 keV)
    S_X = int_spec(E, EN_E_norm, E_min=0.3e3, E_max=10e3)
    S_X = jnp.nan_to_num(S_X, nan=0.0)

    # FRED light curve — computed ONCE
    t = jnp.geomspace(1e-4, 1e2, 1000)
    L = fred(t, 0.1, 0.35)
    S_unit = int_lc(t, L)
    S_unit = jnp.where(jnp.abs(S_unit) < 1e-300, 1e-300, S_unit)

    L_gamma = (S_gamma / S_unit)[..., jnp.newaxis] * L
    L_X = (S_X / S_unit)[..., jnp.newaxis] * L

    return E, EN_E_norm, t, L_gamma, S_gamma, L_X, S_X


def obs_grid(eps, e_iso_grid_arr, amati_a=0.41, amati_b=0.83, e_1=0.3e3, e_2=10e3):
    """Single-band obs_grid (kept for backward compat)."""
    alpha, beta_val = -1, -2.3
    e_iso_grid_arr = jnp.asarray(e_iso_grid_arr)
    E_p = 1e5 * 10**(amati_a * jnp.log10(jnp.maximum(e_iso_grid_arr, 1e-300) / 1e51) + amati_b)
    E_0 = E_p / (2 + alpha)
    E = jnp.geomspace(1e2, 1e8, 1000)
    EN_E = E * band(E, alpha, beta_val, E_0)
    eps_unit = int_spec(E, EN_E, E_min=10e3, E_max=1e6)
    eps_unit = jnp.where(jnp.abs(eps_unit) < 1e-300, 1e-300, eps_unit)
    A_spec = eps / eps_unit
    EN_E_norm = A_spec[..., jnp.newaxis] * EN_E
    S = int_spec(E, EN_E_norm, E_min=e_1, E_max=e_2)
    S = jnp.nan_to_num(S, nan=0.0)
    t = jnp.geomspace(1e-4, 1e2, 1000)
    L = fred(t, 0.1, 0.35)
    S_unit = int_lc(t, L)
    A_lc = S / jnp.where(jnp.abs(S_unit) < 1e-300, 1e-300, S_unit)
    L_scaled = A_lc[..., jnp.newaxis] * L
    return E, EN_E_norm, t, L_scaled, S


# ── Grid construction (NOT JIT'd — uses Python strings) ───────────────────

def coord_grid(n_theta, n_phi, theta_bounds, phi_bounds):
    theta_1d = -jnp.cos(jnp.linspace(theta_bounds[0], theta_bounds[1], n_theta)) * jnp.pi / 2 + jnp.pi / 2
    theta, phi = jnp.meshgrid(theta_1d, jnp.linspace(phi_bounds[0], phi_bounds[1], n_phi), indexing='ij')
    return theta, phi


def gamma_grid(g0, theta, phi, struct='tophat', theta_c=jnp.deg2rad(5.0), cutoff=None, **kwargs):
    if callable(struct):
        g = g0 * struct(theta, phi)
    elif struct == 'tophat':
        g = g0 * jnp.ones_like(theta)
    elif struct == 'gaussian':
        g = g0 * gaussian(theta, theta_c)
    elif struct == 'powerlaw':
        k = kwargs.get('k', 2)
        g = g0 * powerlaw(theta, theta_c, k)
    else:
        raise ValueError("struct must be 'tophat', 'gaussian', 'powerlaw', or callable")

    if cutoff:
        g = jnp.where(jnp.abs(jnp.cos(theta)) < jnp.cos(cutoff), 1.0, g)
    g = jnp.maximum(g, 1.0)
    return g


def eps_grid(eps0, theta, phi, struct='tophat', theta_c=jnp.deg2rad(5.0), cutoff=None, **kwargs):
    if callable(struct):
        eps_north = eps0 * struct(theta, phi)
    elif struct == 'tophat':
        eps_north = eps0 * jnp.ones_like(theta)
    elif struct == 'gaussian':
        eps_north = eps0 * gaussian(theta, theta_c)
    elif struct == 'powerlaw':
        k = kwargs.get('k', 0)
        eps_north = eps0 * powerlaw(theta, theta_c, k)
    else:
        raise ValueError("struct must be 'tophat', 'gaussian', 'powerlaw', or callable")

    theta_mirror = jnp.pi - theta
    if callable(struct):
        eps_south = eps0 * struct(theta_mirror, phi)
    elif struct == 'tophat':
        eps_south = eps0 * jnp.ones_like(theta)
    elif struct == 'gaussian':
        eps_south = eps0 * gaussian(theta_mirror, theta_c)
    elif struct == 'powerlaw':
        k = kwargs.get('k', 0)
        eps_south = eps0 * powerlaw(theta_mirror, theta_c, k)

    eps = jnp.maximum(eps_north, eps_south)
    if cutoff:
        eps = jnp.where(jnp.abs(jnp.cos(theta)) < jnp.cos(cutoff), 0.0, eps)
    return eps


# ── CSV I/O (numpy only) ──────────────────────────────────────────────────

def save_data(jet, wind, theta_los, phi_los, path='./', model_id=0):
    t_np = np.asarray(jet.t)
    Lg_np = np.asarray(jet.L_gamma_tot)
    Lx_np = np.asarray(jet.L_X_tot)
    fname = path + '{}_data.csv'.format(int(round(float(np.rad2deg(theta_los)))))
    with open(fname, mode='w', newline='') as file:
        writer = csv.writer(file)
        if model_id in (1, 2):
            wt_np = np.asarray(wind.engine.t)
            wLx_np = np.asarray(wind.L_X_tot)
            writer.writerow(['jet_t', 'jet_L_gamma', 'jet_L_X', 'wind_t', 'wind_L_X'])
            for i in range(t_np.shape[0]):
                writer.writerow([t_np[i], Lg_np[i], Lx_np[i], wt_np[i], wLx_np[i]])
        elif model_id in (3, 4):
            writer.writerow(['jet_t', 'jet_L_gamma', 'jet_L_X'])
            for i in range(t_np.shape[0]):
                writer.writerow([t_np[i], Lg_np[i], Lx_np[i]])


# ── Misc helpers (exported for jet.py) ─────────────────────────────────────

def _int_energy_1d(E, EN, E1=None, E2=None):
    mask = jnp.ones_like(E)
    if E1 is not None:
        mask = mask * (E >= E1)
    if E2 is not None:
        mask = mask * (E <= E2)
    return jnp.trapezoid(EN * mask, E)


def _interp_1d(x, y, xq):
    return jnp.interp(xq, x, y, left=0.0, right=0.0)
