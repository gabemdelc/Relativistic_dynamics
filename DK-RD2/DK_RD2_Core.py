# coding=utf-8
###################################################################################
# DK-RD2 Core Utilities – Constants, Functions, and Dynamic Gravitational Engine
###################################################################################
#    Author:        GabE=mc² (gabemdelc@gmail.com)
#    Created:       11/Feb/2025
#    Last Revision: Feb/2026
#    Project:       DK-RD2 — DK-Relativistic Dynamics Model (2.0)
#    License:       MIT License
#    Repository:    https://github.com/gabemdelc/Relativistic_dynamics
###################################################################################
"""
====================================================================================
DK_RD2_Core.py — Core Computational Engine (CLASS-free)
====================================================================================

Purpose
-------
Self-contained core implementing DK-RD2 background physics:
• Exact thermo–relativistic coupling Gab(T,v).
• Matter rescaling Ω_m → Ω_m·(Gab/Core_G0).
• Expansion functions E(z)=H/Core_H0 for ΛCDM and DK‑RD2 (no‑Λ by default).
• Distances (χ, D_A, D_L, μ), lensing geometry (D_l, D_s, D_ls; SIS & point-mass θ_E).
• CMB background geometry via the acoustic scale ℓ_A (remapping only).

No Boltzmann solver is imported. For comparisons to theoretical CMB curves,
pass a reference D_ℓ grid (ℓ, Dl_ref) to the remapping helpers below.

Units & conventions
-------------------
• G in SI (m^3 kg^-1 s^-2); c in m/s; Core_H0 in km/s/Mpc; distances in Mpc.
• E(z) is dimensionless; all distance integrals use 1/E(z).
• Default models: T(z)=Core_T0·(1+z), v(z)=0 unless overridden.

Separation of concerns
----------------------
This file is the *core*: constants, physics, and numerical routines.
Dataset I/O, plotting, and figure orchestration should live in the driver.
"""

# =========================
# 0) Imports and constants
# =========================
from typing import Callable, Optional

import pandas as pd
from scipy.integrate import quad as _quad_scipy  # single alias for all integrations
import numpy as np

# Visual reference in figure footers
Core_author = "©2026 GabE=mc² — Gabriel Martín del Campo F."
Core_git_gabe = "https://github.com/gabemdelc/Relativistic_dynamics"
Core_zenodo = "Zenodo DOI: https://doi.org/10.5281/zenodo.18529901"
Core_autor_text = (
    f"{Core_author}."
    "Reproducible from public DK-RD2 code. \n"
    f"GitHub: {Core_git_gabe} | Zenodo: {Core_zenodo}"
)
# Output directory (used by evidence output files)
Core_out_dir_path = "evidence/"

# Universal constants
Core_G0 = Core_G_const = 6.67430e-11           # m^3 kg^-1 s^-2
# Speed of light in vacuum (defined constant, exact)
Core_c_light = 299_792_458       # [m/s]
Core_c_km_s  = 299_792.458       # [km/s]
# --- CMB temperature today
# Core_T0 [K] used in DK-RD2 thermodynamic background
Core_TCMB_K = Core_T_fixed = Core_T0 = 2.7255       # K (Planck 2018)
"""
###################################################################################
# --- DEFAULT ΛCDM COSMOLOGY ---
# PLANCK 2018 BASELINE PARAMETERS USED AS REFERENCE VALUES.
# THESE CONSTANTS ARE EMPLOYED EXCLUSIVELY FOR COMPARISON AGAINST THE ΛCDM MODEL.
# --- ΛCDM COMPARISON BASELINES (PLANCK-LIKE; USED ONLY FOR ΛCDM CURVES) ---
###################################################################################
"""
Core_H0 = Core_Hubble_H0 = 67.7               # Hubble constant today [km/s/Mpc]
Core_Omega_gamma0 = 2.472e-5                  # photon density today for Tcmb = Core_TCMB_K and h = 1
Core_Omega_L = Core_OMEGA_L_LCDM   = 0.685    # Dark energy density parameter (Λ term) Ω_Λ for ΛCDM reference (flat: 1 - Ω_m)
Core_Omega_b = 0.049                          # Baryon density parameter
Core_Omega_m = Core_OMEGA_M_LCDM   = 0.315    # Ω_m for ΛCDM reference
Core_Omega_m_fixed_for_DK = 1.0               # or physically derived from Gab(T,v)
Core_h_Planck = 6.62607015e-34                # Planck constant [J·s]
Core_k_B = 1.380649e-23                       # Boltzmann constant [J/K]
# Use a positive constant to make the code read naturally
Core_M_B_abs = 19.253  # |M_B| from Riess et al. 2022

# ------------------------------------------------------------
# Convert apparent magnitudes (m_B_corr) → distance modulus (μ)
#
# Pantheon+ and earlier SN Ia compilations often provide
# "m_B_corr" (corrected apparent B-band magnitudes) rather than
# distance moduli μ. To compare with theoretical models, we must
# subtract the absolute magnitude of a standard SN Ia:
#
#     μ = m_B_corr - M_B
#
# Here we adopt:
#     M_B = -19.3  (typical absolute magnitude for SNe Ia)
#
# This is NOT an ad-hoc parameter. It is an empirically calibrated
# constant derived from multiple independent studies:
#
#   • Riess et al. 2016, ApJ 826:56 — M_B = −19.25 ± 0.03  (SH0ES calibration)
#   • Betoule et al. 2014, A&A 568:A22 (JLA) — M_B = −19.05  (empirical average)
#   • Scolnic et al. 2018, ApJ 859:101 (Pantheon) — M_B = −19.36 ± 0.02  (full calibration)
#   • Riess et al. 2022, ApJ 934:7 (SH0ES update) — M_B = −19.253 ± 0.027
#
# The adopted value (−19.3) represents a physically meaningful
# average and simply defines the zero-point of the magnitude scale.
# It does NOT alter the physics of the DK-RD² model; it only shifts
# the vertical reference for μ(z).
# ------------------------------------------------------------


# ---------------------------------------------------------------------
# Lensing defaults (used when catalogs do not provide θ_E uncertainties)
# ---------------------------------------------------------------------
# THETA_ERR_FRAC_DEFAULT:
#   Dimensionless fractional error applied to the Einstein radius when
#   a catalog lacks an explicit θ_E uncertainty column. For example,
#   if θ_E = 1.2 arcsec and THETA_ERR_FRAC_DEFAULT = 0.05, we assign
#   σ(θ_E) = 0.05 × 1.2 = 0.06 arcsec.
THETA_ERR_FRAC_DEFAULT: float = 0.05   # 5% heuristic

# THETA_ERR_FLOOR:
#   Absolute (additive) error floor in arcseconds applied when the
#   computed fractional error is unrealistically small or when the
#   catalog value is missing/zero. This avoids σ=0 in χ².
THETA_ERR_FLOOR: float = 1e-3          # arcsec


# --- Planck/CLASS baseline constants (used as defaults across figures) ---
Core_Z_RECOMB         = 1089.0      # Recombination redshift used in CMB geometric mapping
Core_NS_PLANCK        = 0.965       # Scalar spectral index (Planck baseline)
Core_AS_PLANCK        = 2.1e-9      # Primordial scalar amplitude at k=0.05 Mpc^-1
Core_TAU_REIO_PLANCK  = 0.054       # Optical depth to reionization (Planck baseline)
Core_LMAX_PLANCK      = 2500        # Typical upper multipole for Planck TT bandpowers
# Present-day σ8 normalization (Planck-like; used as observational anchor)
Core_SIGMA8_PLANCK_2018 = 0.811

# Unit conversions
Core_pc_to_m =  3.08567758149137e16
Core_Mpc_to_m = 3.08567758149137e22
ARCSEC_PER_RAD: float = 206264.80624709636
Core_arcsec_per_rad = ARCSEC_PER_RAD
RAD_PER_ARCSEC: float = 1.0 / ARCSEC_PER_RAD

##########################################
# Convert to scientific notation
##########################################
def sci_notation(x):
    mantissa, exp = f"{x:.5e}".split("e")
    return float(mantissa), int(exp)

# =========================================
# 1) Thermo–relativistic coupling and Ω_m
# =========================================
def Gab(T, v) -> float:
    """
    Gab(T, v) = Core_G0 * (1 + (v^2/c^2) * (Core_T0/T)).
    Returns SI units (m^3 kg^-1 s^-2). Vectorized for array inputs.
    """
    v = np.clip(np.asarray(v, dtype=float), 0.0, (1.0 - 1e-12) * Core_c_light)
    T = np.maximum(np.asarray(T, dtype=float), 1e-20)
    beta2 = (v / Core_c_light) ** 2
    return Core_G0 * (1.0 + beta2 * (Core_T0 / T))

def Gab_z(z, v_model: Optional[Callable] = None, T_model: Optional[Callable] = None):
    """
    Gab at redshift z with explicit models (defaults: T=Core_T0(1+z), v=0).
    """
    z = np.asarray(z, dtype=float)
    T = (Core_T0 * (1.0 + z)) if T_model is None else np.asarray(T_model(z), dtype=float)
    v = (np.zeros_like(z)) if v_model is None else np.asarray(v_model(z), dtype=float)
    return Gab(T, v)

def Omega_m_Gab(Om_m: float, Gab_value):
    """Ω_m^eff = Ω_m * (Gab / Core_G0)."""
    return Om_m * (np.asarray(Gab_value, dtype=float) / float(Core_G_const))

def Omega_m_ab_z(
    z,
    Omega_m0: float | None = None,
    v_model: Optional[Callable] = None,
    T_model: Optional[Callable] = None,
    Omega_L_value: float | None = None,
):
    """
    Self-consistent matter fraction for DK-RD2:

        Core_Omega_m,ab(z) = Omega_m0 * (1+z)^3 * mu(z) / E(z)^2

    where:
        mu(z) = Gab(z)/Core_G0
        E(z)  = E_Relativistic(z)  (DK background)

    Notes
    -----
    - Omega_m0 is NOT fitted here. If None, uses module-level Core_Omega_m_fixed_for_DK.
    - Omega_L_value is kept for controlled tests (default None => DK closure).
    """
    z = np.asarray(z, dtype=float)

    # If Omega_m0 is not provided, fall back to the DK fixed normalization
    Om0 = float(Core_Omega_m_fixed_for_DK) if Omega_m0 is None else float(Omega_m0)

    # mu(z) = Gab/Core_G0
    mu = np.asarray(Gab_z(z, v_model=v_model, T_model=T_model), dtype=float) / float(Core_G0)

    # DK background E(z)
    Ez = np.asarray(
        E_Relativistic(
            z,
            Core_Omega_m=Om0,
            Omega_L_value=Omega_L_value,
            v_model=v_model,
            T_model=T_model,
        ),
        dtype=float,
    )

    zp1 = 1.0 + z
    Om_ab = Om0 * (zp1 ** 3) * mu / np.clip(Ez ** 2, 1e-300, None)
    return Om_ab

def Omega_m_Gab_z(z, Om_m: float, v_model: Optional[Callable] = None,
                  T_model: Optional[Callable] = None):
    """Ω_m^eff(z) wrapper."""
    return Omega_m_Gab(Om_m, Gab_z(z, v_model=v_model, T_model=T_model))

# =========================================
# 2) Expansion functions E(z) and H(z)
# =========================================

def E_LCDM_unary(
    z,
    Omega_m: float | None = None,
    Omega_L: float | None = None,
    Omega_k: float = 0.0,
    Omega_r: float = 0.0,
):
    """
    Unary wrapper for E_LCDM(z, Ωm, ΩΛ, ...).

    If Omega_m / Omega_L are not provided, it uses the ΛCDM
    reference constants defined in this module (Core_OMEGA_M_LCDM,
    Core_OMEGA_L_LCDM). If they are provided, those override the defaults.
    """
    # Decide which Ωm to use
    if Omega_m is None:
        Om = float(Core_OMEGA_M_LCDM)
    else:
        Om = float(Omega_m)

    # Decide which ΩΛ to use
    if Omega_L is None:
        OL = float(Core_OMEGA_L_LCDM)
    else:
        OL = float(Omega_L)

    return E_LCDM(z, Om, OL, Omega_k=Omega_k, Omega_r=Omega_r)


def E_LCDM(z,
           Core_Omega_m: float,
           Core_Omega_L: float,
           Omega_k: float = 0.0,
           Omega_r: float = 0.0) -> np.ndarray:
    """
    Dimensionless expansion rate E(z) = H(z)/Core_H0 for flat/curved ΛCDM.
    Inputs are density parameters today (Ω_i).
    """
    z = np.asarray(z, dtype=float)
    return np.sqrt(
        float(Omega_r) * (1.0 + z)**4 +
        float(Core_Omega_m) * (1.0 + z)**3 +
        float(Omega_k) * (1.0 + z)**2 +
        float(Core_Omega_L)
    )

def H_LCDM(z,
           Core_H0: float,
           Core_Omega_m: float,
           Core_Omega_L: float,
           Omega_k: float = 0.0,
           Omega_r: float = 0.0) -> np.ndarray:
    """
    H(z) in km/s/Mpc for ΛCDM. Uses *exactly* the Core_H0 provided here.
    """
    Ez = E_LCDM(z, Core_Omega_m, Core_Omega_L, Omega_k=Omega_k, Omega_r=Omega_r)
    return float(Core_H0) * Ez

def E_Relativistic(
    z,
    *_,
    Omega_m: float | None = None,
    Omega_L_value: float | None = None,
    v_model=None,
    T_model=None,
    **extra,
):
    """
    DK-RD2 background expansion E(z) = H(z)/Core_H0.

    Policy:
    - If an explicit Omega_m (or its alias Core_Omega_m) is provided, that value is used.
    - If neither is given, the function falls back to the fixed DK-RD2 normalization
      Core_Omega_m_fixed_for_DK defined at module level (parameter-free choice).
    - If Omega_L_value is None, no explicit Λ term is used; closure is set by today's
      effective matter (Gab at Core_T_fixed, v=0). Passing a float tests a "with-Λ" variant.
    """

    z_arr = np.asarray(z, dtype=float)
    zp1   = 1.0 + z_arr

    # Resolve effective Omega_m
    if Omega_m is None:
        # Allow alias via Core_Omega_m passed in extra kwargs (from comoving_distance, H_Relativistic, etc.)
        if "Core_Omega_m" in extra and extra["Core_Omega_m"] is not None:
            Om_fixed = float(extra["Core_Omega_m"])
        else:
            # Parameter-free DK-RD2 normalization (usually 1.0)
            Om_fixed = float(Core_Omega_m_fixed_for_DK)
    else:
        Om_fixed = float(Omega_m) # use the Parameter passed

    # Neutral baselines if no custom T(z), v(z) are provided
    T_vals = (Core_T_fixed * zp1) if (T_model is None) else np.asarray(T_model(z_arr), dtype=float)
    v_vals = (np.zeros_like(z_arr)) if (v_model is None) else np.asarray(v_model(z_arr), dtype=float)

    # Effective coupling Gab(T,v)
    try:
        Gab_vals = Gab(T_vals, v_vals)
    except Exception:
        Gab_vals = np.vectorize(lambda Ti, vi: Gab(Ti, vi), otypes=[float])(T_vals, v_vals)

    Gab0 = Gab(Core_T_fixed, 0.0)

    # Closure term
    if Omega_L_value is None:
        Omega_closure = 1.0 - Om_fixed * (Gab0 / Core_G0)
    else:
        Omega_closure = float(Omega_L_value)

    # Effective matter at redshift z
    Omega_m_eff_z = Om_fixed * (Gab_vals / Core_G0)

    E2 = (Omega_m_eff_z * (zp1 ** 3)) + Omega_closure
    return np.sqrt(np.clip(E2, 1e-300, None))


def H_Relativistic(
    z,
    Core_H0,
    Core_Omega_m: float | None = None,       # OPTIONAL: if None, use module-level fixed Core_Omega_m
    Omega_L_value: float | None = None,
    v_model=None,
    T_model=None,
    **kwargs
):
    """Physical H(z) in km s^-1 Mpc^-1 from E_Relativistic."""
    import numpy as np
    z_arr = np.asarray(z, dtype=float)
    return float(Core_H0) * E_Relativistic(
        z_arr,
        Core_Omega_m=Core_Omega_m, Omega_L_value=Omega_L_value,
        v_model=v_model, T_model=T_model,
        **kwargs
    )

try:
    from scipy.integrate import cumulative_trapezoid as _cumtrapz
except Exception:
    def _cumtrapz(y, x, initial=0.0):
        y = _np.asarray(y, dtype=float)
        x = _np.asarray(x, dtype=float)
        if y.size != x.size:
            raise ValueError("x and y must have the same length")
        dy = 0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])
        out = _np.cumsum(dy)
        if initial is not None:
            out = _np.concatenate(([float(initial)], out))
        return out

def comoving_distance(
    z,
    E_function,
    Core_H0,
    Core_c_km_s=Core_c_km_s,
    **E_kwargs
):
    """
    χ(z) [Mpc] = (c/Core_H0) ∫_0^z dz'/E(z')
    Robust to scalar/short inputs: integrates on an internal grid [0, z_max] and interpolates.
    """
    z_arr = np.atleast_1d(np.asarray(z, dtype=float))
    z_arr = np.where(np.isfinite(z_arr) & (z_arr >= 0.0), z_arr, 0.0)
    z_max = float(np.nanmax(z_arr)) if z_arr.size else 0.0

    c_over_H0 = Core_c_km_s / float(Core_H0)  # [Mpc]

    if z_max <= 0.0:
        out = np.zeros_like(z_arr, dtype=float)
        return out if out.size > 1 else float(out[0])

    # Internal grid
    n_grid = max(1025, int(2048 * (1.0 + z_max**0.5)))
    z_grid = np.linspace(0.0, z_max, n_grid)

    # Only forward meaningful kwargs to E(z)
    _allowed_E_keys = {
        "Core_Omega_m", "Core_Omega_L", "Omega_k", "Omega_r",        # ΛCDM keys
        "Omega_L_value", "v_model", "T_model"              # DK-RD2 keys
    }
    _E_kwargs = {k: v for k, v in E_kwargs.items() if k in _allowed_E_keys}

    # CRUCIAL: do NOT pass Core_H0 into E_function
    Ez_grid = np.asarray(E_function(z_grid, **_E_kwargs), dtype=float)

    invE  = 1.0 / Ez_grid
    integ = _cumtrapz(invE, z_grid, initial=0.0)  # ∫_0^z dz'/E

    chi_grid = c_over_H0 * integ  # [Mpc]

    chi = np.interp(z_arr, z_grid, chi_grid, left=0.0, right=chi_grid[-1])
    return chi if chi.size > 1 else float(chi[0])

def angular_diameter_distance(
    z,
    E_function,
    Core_H0,
    Core_c_km_s=Core_c_km_s,
    *E_args,
    **E_kwargs
):
    """
    Angular diameter distance D_A(z) [Mpc].

    This is a thin wrapper around comoving_distance(...):

        D_A(z) = χ(z) / (1 + z)

    Parameters
    ----------
    z : float or array-like
        Redshift(s) at which to evaluate the angular diameter distance.
    E_function : callable
        Background expansion function E(z) = H(z)/H0 (e.g. E_LCDM, E_Relativistic).
    Core_H0 : float
        Hubble constant H0 in km/s/Mpc to be used consistently with E_function.
    Core_c_km_s : float, optional
        Speed of light in km/s. Defaults to the module-level Core_c_km_s.
    *E_args, **E_kwargs :
        Extra positional and keyword arguments forwarded to E_function
        (e.g., Core_Omega_m, Core_Omega_L, Omega_L_value, v_model, T_model).

    Notes
    -----
    - This function does NOT hard-code any particular redshift (like recombination).
      Callers must pass the appropriate z (e.g. Core_Z_RECOMB for CMB, z_l/z_s for lensing).
    """
    # Ensure z is an array; comoving_distance already handles arrays vs scalars
    z_arr = np.asarray(z, dtype=float)

    # Comoving distance χ(z) [Mpc]
    Dc = comoving_distance(z_arr, E_function, Core_H0, Core_c_km_s, *E_args, **E_kwargs)

    # Angular diameter distance D_A(z) = χ(z) / (1 + z)
    return Dc / (1.0 + z_arr)

# ==================================================
# Strong-lensing helpers (SIS + angular-diameter combos)
# ==================================================
from typing import Tuple

def D_lens(
    z_l: float,
    z_s: float,
    E_function,
    Core_H0: float,
    Core_c_km_s: float = Core_c_km_s,
    *E_args,
    **E_kwargs
) -> Tuple[float, float, float]:
    """
    Angular-diameter distance trio for lensing geometry (flat FRW):
        D_l  = D_A(0 -> z_l)
        D_s  = D_A(0 -> z_s)
        D_ls = D_A(z_l -> z_s) = (chi(z_s) - chi(z_l)) / (1 + z_s)

    Returns (D_l, D_s, D_ls) in Mpc.

    Notes
    -----
    - Uses the module's comoving_distance() and angular_diameter_distance().
    - Core_c_km_s allows consistent speed-of-light use across callers.
    - **E_kwargs is forwarded to E_function (e.g., Core_Omega_m, Core_Omega_L/Omega_L_value).
    """
    # Comoving distances (Mpc)
    chi_l  = comoving_distance(z_l, E_function, Core_H0, Core_c_km_s, *E_args, **E_kwargs)
    chi_s  = comoving_distance(z_s, E_function, Core_H0, Core_c_km_s, *E_args, **E_kwargs)

    # Angular-diameter distances observer->lens/source (Mpc)
    D_l = chi_l / (1.0 + float(z_l))
    D_s = chi_s / (1.0 + float(z_s))

    # Angular-diameter distance lens->source (Mpc; flat case)
    D_ls = max(0.0, (chi_s - chi_l)) / (1.0 + float(z_s))

    return D_l, D_s, D_ls

def einstein_radius_SIS(
    z_l,
    z_s,
    sigma_v_ms,
    E_function,
    Core_H0,
    Core_c_km_s: float = Core_c_km_s,  # Core_c_km_s: float = 299_792.458
    *E_args,
    **E_kwargs,
):
    """
    SIS Einstein radius (Singular Isothermal Sphere):
        θ_E = 4π (σ_v² / c²) × (D_ls / D_s)

    Parameters
    ----------
    z_l, z_s : float
        Lens and source redshifts (z_s > z_l).
    sigma_v_ms : float
        1D velocity dispersion [m/s].
    E_function : callable
        Expansion function (e.g., E_LCDM or E_Relativistic).
    Core_H0 : float
        Hubble constant [km/s/Mpc].
    Core_c_km_s: float = Core_c_km_s,  # or Core_c_km_s: float = 299_792.458

    Returns
    -------
    (theta_rad, theta_arcsec)
        Einstein radius in radians and arcseconds.
    """
    # Distances in Mpc
    D_l, D_s, D_ls = D_lens(
        z_l, z_s, E_function, Core_H0, Core_c_km_s, *E_args, **E_kwargs
    )

    # Speed of light in m/s
    c_ms = float(Core_c_light)

    # SIS formula
    theta_rad = 4.0 * np.pi * (float(sigma_v_ms) ** 2) / (c_ms ** 2) * (D_ls / max(D_s, 1e-30))

    # Convert to arcseconds
    theta_arcsec = theta_rad * ARCSEC_PER_RAD  # use core constant, no magic number
    # Ensure scalar float outputs
    theta_rad = float(np.atleast_1d(theta_rad)[0])
    theta_arcsec = float(np.atleast_1d(theta_arcsec)[0])

    return theta_rad, theta_arcsec


# --------------------------------------------------------------------------------------
# Luminosity distance + distance modulus for a given expansion function E(z)
# --------------------------------------------------------------------------------------
def luminosity_distance_and_mu(
    z, E_function, Core_H0, Core_c_km_s, *E_args, **E_kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (mu, dL_Mpc). 100% unit-safe. No hidden Ωm for DK branch.
    """

    """
    Luminosity distance and distance modulus for a given expansion function E(z).

    Definitions
    -----------
    - E(z) = H(z)/Core_H0  (dimensionless)
    - χ(z) = (c/Core_H0) ∫_0^z dz'/E(z')            [Mpc]
    - d_L(z) = (1+z) * χ(z)                    [Mpc]
    - μ(z) = 5 log10(d_L/Mpc) + 25             [mag]
    """
    z_arr = _np.atleast_1d(_np.asarray(z, dtype=float))
    z_arr = _np.where(_np.isfinite(z_arr) & (z_arr >= 0.0), z_arr, 0.0)

    # Comoving distance χ(z) [Mpc]
    chi_Mpc = comoving_distance(
        z_arr, E_function, Core_H0, Core_c_km_s=Core_c_km_s, *E_args, **E_kwargs
    )

    # Luminosity distance and μ
    dL_Mpc = (1.0 + z_arr) * chi_Mpc
    with _np.errstate(divide="ignore", invalid="ignore"):
        mu = 5.0 * _np.log10(_np.maximum(dL_Mpc, 1e-99)) + 25.0

    return (mu if mu.size > 1 else float(mu[0])), dL_Mpc

def einstein_radius_point_mass(M_kg: float, z_l: float, z_s: float,
                               E_function: Callable, Core_H0: float,
                               Core_c_km_s,
                               G: float = Core_G0, *E_args, **E_kwargs) -> Tuple[float, float]:
    """
    Point mass lens:
        θ_E = sqrt( (4 G M / c^2) * D_ls / (D_l D_s) ).
    Returns (θ_E in radians, θ_E in arcsec).
    """
    D_l, D_s, D_ls = D_lens(z_l, z_s, E_function, Core_H0, Core_c_km_s, *E_args, **E_kwargs)
    pref = (4.0 * G * float(M_kg)) / (Core_c_light ** 2)  # meters
    D_l_m, D_s_m, D_ls_m = D_l * Core_Mpc_to_m, D_s * Core_Mpc_to_m, D_ls * Core_Mpc_to_m
    theta_rad = np.sqrt(pref * (D_ls_m / (D_l_m * D_s_m)))
    return theta_rad, theta_rad * 206265.0

def einstein_radius(
    z_l: float,
    z_s: float,
    sigma_v_kms: float,
    *,
    E_function=E_LCDM,
    Core_H0: float = Core_Hubble_H0,
    **E_kwargs,
):
    """
    Legacy alias to SIS Einstein radius.
    Returns (theta_rad, theta_arcsec).

    Notes
    -----
    - Uses the module speed-of-light (Core_c_km_s) by default.
    - sigma_v_kms is the 1D velocity dispersion in km/s (SIS). Internally converted to m/s.
    - Extra cosmology params for E_function (e.g., Core_Omega_m, Core_Omega_L/Omega_L_value) are passed via **E_kwargs.
    """
    theta_rad, theta_arcsec = einstein_radius_SIS(
        z_l,
        z_s,
        float(sigma_v_kms) * 1000.0,   # km/s → m/s
        E_function=E_function,
        Core_H0=Core_H0,
        Core_c_km_s=Core_c_km_s,          # use module default
        **E_kwargs,
    )
    return theta_rad, theta_arcsec

# ==================================================
# 5) CMB geometry: ℓ_A, r_s, damping, remaps
# ==================================================
def omega_gamma_from_Tcmb(Tcmb_K: float, h: float) -> float:
    """
    Photon density parameter:
        Ω_γ = Ω_γ0 * (Tcmb / Core_TCMB_K)^4 / h^2

    where Ω_γ0 = 2.472e-5 is the photon density today for Tcmb = Core_TCMB_K and h = 1.

    """

    T0 = float(Core_TCMB_K)
    return Core_Omega_gamma0 * (Tcmb_K / T0) ** 4 / (h ** 2)

def sound_horizon_rs(
    z_star: float,
    E_function: Callable,
    Core_H0: float,
    Core_Omega_b: float,
    Omega_gamma: float,
    Core_c_km_s: float = Core_c_km_s,
    *E_args, **E_kwargs
) -> float:
    """
    r_s = ∫_{z_*}^{∞} [ c_s(z) / H(z) ] dz,
    with c_s = c / sqrt(3*(1+R)),  R(z) = 3Ω_b/(4Ω_γ) * 1/(1+z).
    Returns Mpc.
    """
    c_use = float(Core_c_km_s)  # keep everything in km/s

    def c_over_H(z: float) -> float:
        Ez = float(E_function(float(z), *E_args, **E_kwargs))
        Hz = float(Core_H0) * Ez                        # km/s/Mpc
        Rz = 0.75 * (float(Core_Omega_b) / float(Omega_gamma)) / (1.0 + float(z))
        cs = c_use / np.sqrt(3.0 * (1.0 + Rz))     # km/s
        return float(cs / Hz)                      # Mpc

    z_max = 1.0e5
    res = _quad_scipy(
        c_over_H, float(z_star), float(z_max),
        epsabs=1e-8, epsrel=5e-6, limit=600
    )

    # Accept either (val, err) or just val
    val = float(res[0]) if isinstance(res, (tuple, list)) else float(res)
    return val

def acoustic_scale_ellA(
    z_star: float,
    E_function: Callable,
    Core_H0: float,
    Core_Omega_b: float,
    Omega_gamma: float,
    *E_args, **E_kwargs
) -> float:
    """
    Acoustic scale:
        ℓ_A = π D_A(z_*) / r_s(z_*).
    Returns a scalar float.
    """
    D_A = float(angular_diameter_distance(z_star, E_function, Core_H0))  # enforce scalar
    r_s = float(sound_horizon_rs(
        z_star, E_function, Core_H0, Core_Omega_b, Omega_gamma, Core_c_km_s, *E_args, **E_kwargs
    ))
    return float(np.pi * (D_A / r_s))

# Decoupling redshift (Eisenstein & Hu 1998)
def z_star_EH98(Ob_h2: float, Om_h2: float) -> float:
    """Analytic fit for z_* (decoupling redshift)."""
    b1 = 0.313 * (Ob_h2)**(-0.419) * (1 + 0.607 * (Ob_h2)**0.674)
    b2 = 0.238 * (Ob_h2)**0.223
    return 1048.0 * (1 + 0.00124 * (Ob_h2)**(-0.738)) * (1 + b1 * (Om_h2)**b2)

def R_of_z(z: np.ndarray, Ob_h2: float, Core_TCMB_K: float = 2.7255) -> np.ndarray:
    """
    Baryon–photon inertia ratio R(z)=3ρ_b/(4ρ_γ).
    Uses ρ_γ ∝ T_CMB^4; standard prefactor condensed into 31.5*Ob*h^2*(1e3/z).
    """
    z = np.asarray(z, dtype=float)
    return 31.5 * Ob_h2 * (1.0e3 / np.maximum(z, 1e-8))

def sound_speed(z: np.ndarray, Ob_h2: float) -> np.ndarray:
    """Photon-baryon sound speed:
        c_s(z) = c / sqrt[3(1+R(z))].
    """
    Rz = R_of_z(z, Ob_h2)
    return Core_c_light / np.sqrt(3.0 * (1.0 + Rz))

def r_s_integral(z_star: float, E_func, Core_H0: float, *, Ob_h2: float, **Ekw) -> float:
    """
    Sound horizon at last scattering:
        r_s(z_*) = ∫_{z_*}^{∞} c_s(z)/H(z) dz    [Mpc].
    Uses DK/Λ background via E_func (CLASS-free proxy for distance priors).
    """
    def integrand(z: float) -> float:
        cs = float(sound_speed(z, Ob_h2))             # km/s
        Ez = float(E_func(z, **Ekw))                  # dimensionless E(z)
        Hz = float(Core_H0) * Ez                            # km/s/Mpc
        return cs / Hz                                 # Mpc per unit z

    zmax = 1.0e5
    res = _quad_scipy(integrand, float(z_star), float(zmax),
                      epsabs=1e-7, epsrel=1e-6, limit=300)

    # Accept either (val, err) or just val
    val = float(res[0]) if isinstance(res, (tuple, list)) else float(res)
    return val


def rD_silk_approx(z_star: float, E_func, Core_H0: float, *, Ob_h2: float, **Ekw) -> float:
    """
    Diffusion (Silk) damping scale r_D at z_* (simple approximation).
    Power-law proxy with baryon-loading dependence absorbed; adequate for tail morphing.
    """
    Ez = float(E_func(z_star, **Ekw))
    A = 12.5  # Mpc; coarse normalization near ΛCDM values
    return A * (Ez / 1.0)**(-0.5) * (1.0 + z_star)**(-1.5) * (Ob_h2**(-0.25))

def damp_envelope_ratio(ell: np.ndarray, lD_LCDM: float, lD_DK: float, m: float = 1.25) -> np.ndarray:
    """
    Multiplicative envelope to move the Silk damping tail from lD_LCDM to lD_DK.
    """
    x = np.asarray(ell, dtype=float)
    return np.exp(- (x / lD_DK)**m + (x / lD_LCDM)**m)

def remap_morph_template(
    ell: np.ndarray, ell_ref: np.ndarray, Dl_ref: np.ndarray,
    *, Core_H0: float, Core_Omega_m: float, Core_Omega_L: float,
    Ob_h2: float,  # physical baryon density (e.g., ≈ 0.0224)
    z_star: float | None = None,
) -> tuple[np.ndarray, dict]:
    """
    CLASS-free template morphing:
      1) Shift by the acoustic angle θ_s (depends on r_s and D_A).
      2) Adjust Silk damping tail via l_D ∝ D_A / r_D.

    IMPORTANT:
      - ΛCDM branch uses the provided (Core_Omega_m, Core_Omega_L).
      - DK-RD2 branch MUST NOT receive an external Core_Omega_m. We pass Core_Omega_m=None
        (and Omega_L_value=None) so DK-RD2 uses its internal Gab(T,v) dynamics.
    """
    h = float(Core_H0) / 100.0
    Om_h2 = float(Core_Omega_m) * h**2
    if z_star is None:
        z_star = z_star_EH98(Ob_h2, Om_h2)

    # --- Distances and sound horizons ---
    # ΛCDM (explicit Ωm, ΩΛ)

    DA_LCDM = angular_diameter_distance(
        z_star, E_LCDM, Core_H0, Core_c_km_s,
        Core_Omega_m=Core_Omega_m, Core_Omega_L=Core_Omega_L)

    rs_LCDM = r_s_integral(
        z_star, E_LCDM, Core_H0, Ob_h2=Ob_h2, Core_Omega_m=Core_Omega_m, Core_Omega_L=Core_Omega_L
    )
    rD_LCDM = rD_silk_approx(
        z_star, E_LCDM, Core_H0, Ob_h2=Ob_h2, Core_Omega_m=Core_Omega_m, Core_Omega_L=Core_Omega_L
    )

    # DK-RD2 (NO external Ωm/ΩΛ: let the core decide)
    DA_DK = angular_diameter_distance(
        z_star, E_Relativistic, Core_H0, Core_c_km_s,
        Core_Omega_m=None, Omega_L_value=None)
    rs_DK = r_s_integral(
        z_star, E_Relativistic, Core_H0, Ob_h2=Ob_h2, Core_Omega_m=None, Omega_L_value=None
    )
    rD_DK = rD_silk_approx(
        z_star, E_Relativistic, Core_H0, Ob_h2=Ob_h2, Core_Omega_m=None, Omega_L_value=None
    )

    # --- Acoustic angle ratio ---
    theta_LCDM = rs_LCDM / DA_LCDM
    theta_DK   = rs_DK   / DA_DK
    s_theta    = float(theta_LCDM / theta_DK)  # peak-position scaling

    # --- Diffusion (Silk) multipoles ---
    lD_LCDM = DA_LCDM / rD_LCDM
    lD_DK   = DA_DK   / rD_DK

    # --- 1) θ_s morph (horizontal remap) ---
    ell     = np.asarray(ell, dtype=float)
    ell_ref = np.asarray(ell_ref, dtype=float)
    Dl_ref  = np.asarray(Dl_ref, dtype=float)
    lprime  = np.clip(ell * s_theta, ell_ref.min(), ell_ref.max())
    Dl_morphed = np.interp(lprime, ell_ref, Dl_ref, left=0.0, right=0.0)

    # --- 2) Silk damping envelope ---
    env    = damp_envelope_ratio(ell, float(lD_LCDM), float(lD_DK), m=1.25)
    Dl_out = Dl_morphed * env

    meta = dict(
        z_star=float(z_star),
        DA_LCDM=float(DA_LCDM), DA_DK=float(DA_DK),
        rs_LCDM=float(rs_LCDM), rs_DK=float(rs_DK),
        theta_ratio=float(s_theta),
        lD_LCDM=float(lD_LCDM), lD_DK=float(lD_DK),
    )
    return Dl_out, meta

def Dl_Relativistic(ell: np.ndarray,
                    Dl_ref: np.ndarray,
                    E_relativistic: Callable, H0_rel: float,
                    Omega_m_rel: float, Omega_b_rel: float, Tcmb_rel: float = Core_T_fixed,
                    E_ref: Optional[Callable] = None, H0_ref: Optional[float] = None,
                    Omega_m_ref: Optional[float] = None, Omega_b_ref: Optional[float] = None, Tcmb_ref: float = Core_T_fixed,
                    z_star: float = 1089.0,
                    v_model: Optional[Callable] = None, T_model: Optional[Callable] = None,
                    **E_rel_kwargs) -> np.ndarray:
    """
    Geometry-consistent remapping of a provided reference D_ℓ:
        D_ℓ^DK(ℓ) ≈ D_ℓ^ref( ℓ * (ℓ_A^ref / ℓ_A^DK) ).
    No amplitude rescaling; pure horizontal remap by ℓ_A.
    """
    ell = np.asarray(ell, dtype=float)
    Dl_ref = np.asarray(Dl_ref, dtype=float)
    if Dl_ref.shape != ell.shape:
        raise ValueError("Dl_ref must have the same shape as ell.")

    # DK-RD2 acoustic scale
    h_rel = float(H0_rel) / 100.0
    Omega_gamma_rel = omega_gamma_from_Tcmb(Tcmb_rel, h_rel)
    ellA_rel = acoustic_scale_ellA(
        z_star, E_relativistic, H0_rel, Omega_b_rel, Omega_gamma_rel,
        Omega_m_rel, v_model=v_model, T_model=T_model, **E_rel_kwargs
    )

    # Reference acoustic scale (if not provided, assume same)
    if E_ref is None or H0_ref is None or Omega_m_ref is None or Omega_b_ref is None:
        ellA_ref = ellA_rel
    else:
        h_ref = float(H0_ref) / 100.0
        Omega_gamma_ref = omega_gamma_from_Tcmb(Tcmb_ref, h_ref)
        ellA_ref = acoustic_scale_ellA(z_star, E_ref, H0_ref, Omega_b_ref, Omega_gamma_ref, Omega_m_ref)

    alpha = ellA_ref / ellA_rel
    ell_ref_needed = ell * alpha
    return np.interp(ell_ref_needed, ell, Dl_ref, left=0.0, right=0.0)

def Dl_Relativistic_Gab(*args, **kwargs) -> np.ndarray:
    """Compatibility wrapper → calls Dl_Relativistic()."""
    return Dl_Relativistic(*args, **kwargs)

def remap_geometry_only(
    ell: np.ndarray,
    ell_ref: np.ndarray,
    Dl_ref: np.ndarray,
    *,
    z_star: float = 1089.0,
    Core_H0: float | None = None,
    Core_Omega_m: float | None = None,
    Core_Omega_L: float | None = None,
) -> tuple[np.ndarray, float, float, float]:
    """
    Geometry-only acoustic-scale remapping of a reference CMB D_ℓ(ℓ) curve (CLASS-free).
    Useful as a quick distance-prior morphing without a Boltzmann solver.
    """
    # Resolve cosmological defaults
    H0_use = float(Core_H0)
    Om_m_use = float(Core_Omega_m)
    Om_L_use = float(Core_Omega_L)

    if not np.isfinite(H0_use):
        raise RuntimeError("remap_geometry_only: Core_H0 is None and no module-level Core_H0/Core_Hubble_H0 was found.")
    if not np.isfinite(Om_m_use):
        raise RuntimeError("remap_geometry_only: Core_Omega_m is None and no module-level Core_Omega_m was found.")
    if not np.isfinite(Om_L_use):
        raise RuntimeError("remap_geometry_only: Core_Omega_L is None and no module-level Core_Omega_L was found.")

    # D_A at z_* for both backgrounds
    DA_LCDM = angular_diameter_distance(z_star, E_LCDM, Core_H0=H0_use, Core_Omega_m=Om_m_use, Core_Omega_L=Om_L_use)
    DA_DK = angular_diameter_distance(
        z_star, E_Relativistic, Core_H0=Core_H0, Core_Omega_m=None, Omega_L_value=None
    )

    # Multipole scaling ℓ → ℓ * (DA_LCDM / DA_DK)
    scale = float(DA_LCDM / DA_DK)
    ell     = np.asarray(ell, dtype=float)
    ell_ref = np.asarray(ell_ref, dtype=float)
    Dl_ref  = np.asarray(Dl_ref, dtype=float)
    lprime  = np.clip(ell * scale, ell_ref.min(), ell_ref.max())
    Dl_dk   = np.interp(lprime, ell_ref, Dl_ref, left=0.0, right=0.0)
    return Dl_dk, scale, float(DA_LCDM), float(DA_DK)

# ==================================================
# 6) Metrics utilities and evidence helpers
# ==================================================
def compute_model_metrics(chi2: float, k: int, n: int):
    """Return (AIC, BIC) with honest parameter count k."""
    aic = chi2 + 2 * k
    bic = chi2 + k * np.log(n)
    return aic, bic

# Backward-compatibility alias
compute_modelMetrics = compute_model_metrics  # noqa: N816

def extract_stats(entry, expected: int = 5):
    """
    Normalize a sequence of stats to fixed length.

    Parameters
    ----------
    entry : tuple | list | any
        Sequence with up to `expected` numeric values.
    expected : int
        Desired output length.

    Returns
    -------
    tuple
        First `expected` items (padded with zeros if shorter).
    """
    if isinstance(entry, (tuple, list)):
        return tuple(entry[:expected]) + (0.0,) * max(0, expected - len(entry))
    return (0.0,) * expected

def generate_evidence(evidence_type, consecutive=None, ext="", out_dir=Core_out_dir_path):
    """
    Standardized filenames for saving results (tables, plots, etc.).
    """
    import inspect

    caller_file = inspect.stack()[1].filename
    program_name = os.path.basename(caller_file).replace('.py', '')

    if evidence_type in ("graph", "image"):
        extension = "png"
    elif evidence_type == "table":
        extension = "csv"
    elif evidence_type == "data":
        extension = "txt"
    elif evidence_type == "json":
        extension = "json"
    else:
        extension = ext or "dat"

    prefix = f"{program_name}_{evidence_type}" if evidence_type else program_name
    if consecutive is not None:
        consecutive_str = str(consecutive)

        # If purely numeric → zero-pad
        if consecutive_str.isdigit():
            consecutive_fmt = consecutive_str.zfill(2)
        else:
            # If alphanumeric or decimal (e.g. 7a, 7.1) → keep as-is but pad base number
            import re
            match = re.match(r"(\d+)(.*)", consecutive_str)
            if match:
                base, suffix = match.groups()
                consecutive_fmt = base.zfill(2) + suffix
            else:
                consecutive_fmt = consecutive_str

        file_name = f"{out_dir}{prefix}_{consecutive_fmt}.{extension}"
    else:
        file_name = f"{out_dir}{prefix}.{extension}"
    return file_name

# =========================
# DK-RD2 unified CSV writer
# =========================

import os
import datetime as _dt
import numpy as _np
import pandas as _pd

def _dkrd2_now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")

def dkrd2_to_csv(
    df: _pd.DataFrame,
    filepath: str,
    *,
    table_kind: str = "DEFAULT",
    figure_id: str | int | None = None,
    strict: bool = False,
    index: bool = False,
    float_format: str = "%.10g",
    na_rep: str = "NaN",
    meta: dict | None = None,
) -> str:
    """
    Standardized DK-RD2 CSV writer with reproducible metadata header.

    - Creates folder automatically
    - Writes a comment header (lines starting with '#')
    - Optional schema enforcement by 'table_kind'
    - Consistent float formatting / NaN representation

    Returns filepath.
    """
    # --- defensive ---
    if df is None:
        raise ValueError("dkrd2_to_csv: df is None")
    if not isinstance(df, _pd.DataFrame):
        df = _pd.DataFrame(df)

    # --- folder ---
    folder = os.path.dirname(filepath)

    # --- optional schema rules (extend as you wish) ---
    REQUIRED = {
        "SN_MU": {"z", "mu_obs", "mu_err", "mu_LCDM", "mu_DKRD2"},
        "CMB_TT": {"ell", "Dl_obs", "Dl_err"},
        "HZ": {"z", "Hz_obs", "Hz_err"},
        "LENS_SIS": {"z_l", "z_s"},
        "DEFAULT": set(),
    }

    if strict:
        req = REQUIRED.get(table_kind, set())
        missing = sorted([c for c in req if c not in df.columns])
        if missing:
            raise ValueError(f"dkrd2_to_csv(strict): missing columns for {table_kind}: {missing}")

    # --- header metadata ---
    header = []
    header.append("# DK-RD2 evidence table")
    header.append(f"# created: {_dkrd2_now_iso()}")
    if figure_id is not None:
        header.append(f"# figure: {figure_id}")
    header.append(f"# table_kind: {table_kind}")
    # If you keep these constants in core, they will be available here:
    try:
        header.append(f"# repository: {Core_git_gabe}")
    except Exception:
        pass
    try:
        header.append(f"# author: {Core_author}")
    except Exception:
        pass

    if meta:
        for k, v in meta.items():
            header.append(f"# meta_{k}: {v}")

    # --- write file: header + CSV body ---
    with open(filepath, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(header) + "\n")
        df.to_csv(
            f,
            index=index,
            float_format=float_format,
            na_rep=na_rep,
        )

    return filepath

# Generic CSV STATS writers
def dkrd2_stats_to_csv(
    stats_df: _pd.DataFrame,
    filepath: str,
    *,
    figure_id: str | int | None = None,
    fit_mode: str | None = None,
    index: bool = False,
    strict: bool = False,
    meta: dict | None = None,
) -> str:
    """
    Standardized stats CSV. Keeps numeric columns numeric (AIC/BIC should be NaN if not applicable).
    """
    if not isinstance(stats_df, _pd.DataFrame):
        stats_df = _pd.DataFrame(stats_df)

    if fit_mode is not None and "fit_mode" not in stats_df.columns:
        stats_df = stats_df.copy()
        stats_df["fit_mode"] = fit_mode

    # Minimal recommended schema (extend as you wish)
    REQUIRED = {"model"}
    if strict:
        missing = sorted([c for c in REQUIRED if c not in stats_df.columns])
        if missing:
            raise ValueError(f"dkrd2_stats_to_csv(strict): missing columns: {missing}")

    return dkrd2_to_csv(
        stats_df,
        filepath,
        table_kind="STATS",
        figure_id=figure_id,
        strict=False,
        index=index,
        meta=meta,
    )

# ==================================================
# 7) I/O helpers for example datasets
# ==================================================
def load_planck_tt(path):
    """
    Load Planck TT bandpowers from a whitespace-delimited table.
    Accepts either:
      (ℓ, Dℓ, -dDℓ, +dDℓ)   → four columns
      (ℓ, Dℓ, σ)             → three columns

    Returns:
        ell    : multipoles ℓ (array, float)
        Dl_obs : observed bandpowers Dℓ(ℓ) (array, float)
        Dl_err : symmetric 1σ error bars (array, float)
    """
    cmb = pd.read_csv(path, sep=r"\s+", comment="#", header=None)
    ncol = cmb.shape[1]
    if ncol >= 4:
        cmb = cmb.iloc[:, :4]
        cmb.columns = ["ell", "Dl_obs", "dDl_minus", "dDl_plus"]
        Dl_err = 0.5 * (np.abs(cmb["dDl_minus"].values) + np.abs(cmb["dDl_plus"].values))
    elif ncol == 3:
        cmb.columns = ["ell", "Dl_obs", "Dl_err"]
        Dl_err = np.abs(cmb["Dl_err"].values)
    else:
        raise ValueError("CMB file must have 3 or 4 columns: ℓ, Dℓ, (−err, +err) or (σ).")

    cmb = cmb.replace([np.inf, -np.inf], np.nan).dropna(subset=["ell", "Dl_obs"])
    ell = cmb["ell"].astype(float).values
    Dl_obs = cmb["Dl_obs"].astype(float).values
    Dl_err = np.where(np.isfinite(Dl_err) & (Dl_err > 0), Dl_err, np.maximum(1e-3, 0.05 * np.abs(Dl_obs)))
    return ell, Dl_obs, Dl_err

def build_data_reference(ell, Dl_obs, window=9):
    """
    Build a smoothed data-derived reference spectrum Dℓ(ℓ).
    Uses a simple moving-average with reflective padding to preserve edges.
    """
    w = int(max(3, window))
    pad = w // 2
    x = np.pad(Dl_obs, (pad, pad), mode="reflect")
    kernel = np.ones(w) / w
    smooth = np.convolve(x, kernel, mode="valid")
    return np.asarray(ell, float), np.asarray(smooth, float)


def normalize_lensing_catalog(path: str):
    """
    Read a lensing catalog (CSV/XLSX/TXT) and return standardized columns:
    z_l, z_s, sigma_v_kms, theta_obs_arcsec, theta_err_arcsec

    CASTLES-aware: parses values like '(0.93)', '0.09/0.62', '1.14/3.93?'.
    """
    import pathlib, re
    import numpy as np
    import pandas as pd

    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # --- read ---
    ext = p.suffix.lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(p, sheet_name=0, engine=None)
    else:
        # Try auto-sep; fallback to multi-space; fallback to whitespace
        try:
            df = pd.read_csv(p, engine="python", sep=None, comment="#")
            if df.shape[1] == 1:
                raise ValueError("Single column detected; retry whitespace parse.")
        except Exception:
            try:
                df = pd.read_csv(p, engine="python", sep=r"\s{2,}", comment="#")
                if df.shape[1] == 1:
                    raise ValueError("Still single column; retry \\s+")
            except Exception:
                df = pd.read_csv(p, engine="python", sep=r"\s+", comment="#", header=0)

    if df.empty:
        raise ValueError("Empty file.")

    # --- standardize header names ---
    def clean_name(c: str) -> str:
        c2 = str(c).strip().lower()
        c2 = re.sub(r"[^\w]+", "", c2)
        return c2

    df.columns = [clean_name(c) for c in df.columns]

    # --- CASTLES numeric parser ---
    def _parse_castles_value(x):
        if x is None:
            return np.nan
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none", "---"}:
            return np.nan
        s = s.replace("(", "").replace(")", "").replace("?", "").strip()
        if "/" in s:
            s = s.split("/")[0].strip()   # policy: take first
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        return float(m.group(0)) if m else np.nan

    def to_num(series: pd.Series) -> np.ndarray:
        """
        Smart numeric cast:
        - if already numeric -> fast path
        - else parse as CASTLES-like text
        """
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors="coerce").values
        return series.apply(_parse_castles_value).astype(float).values

    # --- alias maps (add CASTLES keys!) ---
    alias = {
        "z_l": ["zl", "zlens", "zlensz", "zlensredshift", "zlensgalaxy", "zlensgal", "z_l"],
        "z_s": ["zs", "zsource", "zsrc", "zsourceredshift", "z_s"],
        "sigma_v_kms": ["sigmav", "sigma", "sigmakms", "velocitydispersion", "sigma_v_kms"],
        "theta_obs_arcsec": ["size", "sizearcsec", "thetae", "thetaobsarcsec", "thetaearcsec", "theta_obs_arcsec"],
        "theta_err_arcsec": ["thetaerr", "thetaeerr", "theta_err_arcsec"],
    }

    def pick(colkeys):
        for k in colkeys:
            if k in df.columns:
                return k
        return None

    def pick_from_alias(key):
        if key in df.columns:
            return key
        return pick(alias.get(key, []))

    col_zl = pick_from_alias("z_l")
    col_zs = pick_from_alias("z_s")
    col_sv = pick_from_alias("sigma_v_kms")
    col_th = pick_from_alias("theta_obs_arcsec")
    col_te = pick_from_alias("theta_err_arcsec")

    out = pd.DataFrame()
    out["z_l"] = to_num(df[col_zl]) if col_zl is not None else np.nan
    out["z_s"] = to_num(df[col_zs]) if col_zs is not None else np.nan
    out["sigma_v_kms"] = to_num(df[col_sv]) if col_sv is not None else np.nan
    out["theta_obs_arcsec"] = to_num(df[col_th]) if col_th is not None else np.nan

    if col_te is not None:
        out["theta_err_arcsec"] = to_num(df[col_te])
    else:
        out["theta_err_arcsec"] = np.nan

    # Strip obvious non-physical values
    for k in ("z_l", "z_s"):
        out.loc[(out[k] < 0) | (~np.isfinite(out[k])), k] = np.nan
    out.loc[(out["sigma_v_kms"] <= 0) | (~np.isfinite(out["sigma_v_kms"])), "sigma_v_kms"] = np.nan
    out.loc[(out["theta_obs_arcsec"] <= 0) | (~np.isfinite(out["theta_obs_arcsec"])), "theta_obs_arcsec"] = np.nan

    return out

# ==================================================
# 8) Legacy wrapper (μ only) for convenience
# ==================================================
def luminosity_distance(
    z, E_function, Core_H0, Core_c_km_s, *E_args, **E_kwargs
) -> np.ndarray:
    """
    d_L(z) = (1+z) * χ(z), with χ computed by comoving_distance(). Returns Mpc.
    """
    # ΛCDM path: guarantee both Ωm and ΩΛ; keep flat if only Ωm provided.
    if E_function is E_LCDM:
        Om_m = E_kwargs.get("Core_Omega_m", Core_Omega_m)
        Om_L = E_kwargs.get("Core_Omega_L", None)
        if Om_L is None:
            Om_L = 1.0 - float(Om_m)
        E_kwargs["Core_Omega_m"] = float(Om_m)
        E_kwargs["Core_Omega_L"] = float(Om_L)

    # DK-RD2 path: do NOT inject Ωm/Λ defaults here; leave kwargs as-is.

    chi_Mpc = comoving_distance(
        z, E_function, Core_H0, Core_c_km_s, *E_args, **E_kwargs
    )
    z_arr = np.asarray(z, dtype=float)
    return (1.0 + z_arr) * np.asarray(chi_Mpc, dtype=float)

def luminosity_distance_LCDM(
    z,
    Core_Omega_m: float | None = None,
    Core_Omega_L: float | None = None,
    Core_H0: float | None = None,
):
    """
    ΛCDM luminosity distance using module-level cosmological constants.

    If any of Core_Omega_m, Core_Omega_L or Core_H0 are not provided,
    the function falls back to the global module-level values:
        - Core_Omega_m
        - Core_Omega_L
        - Core_H0

    This keeps the routine parameter-free by default, while still allowing
    explicit overrides for controlled tests.
    """
    # Resolve effective parameters from globals if not explicitly passed
    if Core_Omega_m is None:
        Core_Omega_m = float(globals()["Core_Omega_m"])
    if Core_Omega_L is None:
        Core_Omega_L = float(globals()["Core_Omega_L"])
    if Core_H0 is None:
        Core_H0 = float(globals()["Core_H0"])

    return luminosity_distance(
        z,
        E_LCDM,
        Core_H0=Core_H0,
        Core_c_km_s=Core_c_km_s,
        Core_Omega_m=Core_Omega_m,
        Core_Omega_L=Core_Omega_L,
    )

# --------------------------------------------------------------------------------------
# DK-RD2 convenience wrapper — NO external Ωm injection by default
# --------------------------------------------------------------------------------------
def luminosity_distance_Relativistic_temp(
    z: float | np.ndarray,
    Core_Omega_m: float | None = None,          # ← default None so DK uses its internal Ωm
    Core_H0: float = Core_Hubble_H0,
    Omega_L_value: float | None = None,    # DK has no Λ by default; pass a float to test with-Λ
    v_model=None,
    T_model=None
) -> np.ndarray:
    """
    Convenience wrapper for DK-RD2 luminosity distance with thermodynamic coupling.

    Policy
    ------
    • Do NOT inject an external Ωm by default (Core_Omega_m=None). DK-RD2 uses its
      internal fixed matter normalization unless you explicitly override it.
    • No Λ by default (Omega_L_value=None). Provide a float to test a with-Λ variant.

    Returns
    -------
    d_L(z) [Mpc] as a 1-D numpy array (even for scalar z).
    """
    _, dL = luminosity_distance_and_mu(
        z,
        E_Relativistic,
        Core_H0=Core_H0,
        Core_c_km_s =Core_c_km_s,       # use the constant defined in _Core.py
        Core_Omega_m=Core_Omega_m,              # ← stays None unless caller sets it
        Omega_L_value=Omega_L_value,
        v_model=v_model,
        T_model=T_model
    )
    return dL

# =============================================================================
# Linear growth factor D(z) and fσ8(z) for DK-RD2 and ΛCDM
# =============================================================================

def growth_factor_DKRD2_ODE(
    z_array,
    v_model: Optional[Callable] = None,
    T_model: Optional[Callable] = None,
    Omega_m0: float | None = None,
    Omega_L_value: float | None = None,
    a_min: float = 1e-3,
    n_grid: int = 6000,
    use_mu_squared: bool = True,
):
    """
    Linear growth factor D(z) in DK-RD2, computed from the second-order ODE in x = ln(a):

        D_xx + [2 + (d ln H / d x)] D_x  - (3/2) * S(x) * D = 0

    with:
        x = ln(a),  a = e^x,  z = e^{-x} - 1
        H(z) = Core_H0 * E_Relativistic(z)
        mu(z) = Gab(z)/Core_G0
        Core_Omega_m,ab(z) = Omega_m0*(1+z)^3*mu(z)/E(z)^2

    Source term:
        If use_mu_squared=True:
            S(x) = Core_Omega_m,ab(z) * mu(z)   -> (mu^2 / E^2) behavior (fully consistent with modified Poisson)
        Else:
            S(x) = Core_Omega_m,ab(z)           -> (mu / E^2) behavior (if you want a conservative variant)

    Initial conditions (deep matter era approximation):
        D(a) ∝ a  =>  D(x_i)=a_min,  D_x(x_i)=D(x_i)

    Output:
        D(z) normalized to D(0)=1.
    """
    z_array = np.asarray(z_array, dtype=float)

    Om0 = float(Core_Omega_m_fixed_for_DK) if Omega_m0 is None else float(Omega_m0)

    # Integration grid in x = ln(a)
    a_min = float(max(a_min, 1e-8))
    x_i = np.log(a_min)
    x_f = 0.0
    x_grid = np.linspace(x_i, x_f, int(n_grid))
    a_grid = np.exp(x_grid)
    z_grid = (1.0 / a_grid) - 1.0

    # Background E(z) and mu(z)
    Ez = np.asarray(
        E_Relativistic(
            z_grid,
            Core_Omega_m=Om0,
            Omega_L_value=Omega_L_value,
            v_model=v_model,
            T_model=T_model,
        ),
        dtype=float,
    )
    mu = np.asarray(Gab_z(z_grid, v_model=v_model, T_model=T_model), dtype=float) / float(Core_G0)

    # d ln H / d x = d ln E / d x (Core_H0 cancels)
    lnE = np.log(np.clip(Ez, 1e-300, None))
    dlnH_dx = np.gradient(lnE, x_grid)

    Om_ab = Omega_m_ab_z(
        z_grid,
        Omega_m0=Om0,
        v_model=v_model,
        T_model=T_model,
        Omega_L_value=Omega_L_value,
    )

    # Source term S(x)
    if use_mu_squared:
        S = Om_ab * mu
    else:
        S = Om_ab

    # ODE system:
    # y0 = D, y1 = dD/dx
    # y0' = y1
    # y1' = -[2 + dlnH_dx]*y1 + (3/2)*S*y0
    def rhs(i, y0, y1):
        A = 2.0 + float(dlnH_dx[i])
        B = 1.5 * float(S[i])
        dy0 = y1
        dy1 = -A * y1 + B * y0
        return dy0, dy1

    # RK4 integrate on uniform x grid
    D = np.zeros_like(x_grid, dtype=float)
    Dp = np.zeros_like(x_grid, dtype=float)

    # ICs: D ~ a, dD/dx = D
    D[0] = a_min
    Dp[0] = D[0]

    dx = float(x_grid[1] - x_grid[0])

    for i in range(len(x_grid) - 1):
        y0, y1 = D[i], Dp[i]

        k1_0, k1_1 = rhs(i, y0, y1)
        k2_0, k2_1 = rhs(i, y0 + 0.5 * dx * k1_0, y1 + 0.5 * dx * k1_1)
        k3_0, k3_1 = rhs(i, y0 + 0.5 * dx * k2_0, y1 + 0.5 * dx * k2_1)
        k4_0, k4_1 = rhs(i, y0 + dx * k3_0, y1 + dx * k3_1)

        D[i + 1] = y0 + (dx / 6.0) * (k1_0 + 2.0 * k2_0 + 2.0 * k3_0 + k4_0)
        Dp[i + 1] = y1 + (dx / 6.0) * (k1_1 + 2.0 * k2_1 + 2.0 * k3_1 + k4_1)

    # Normalize so D(z=0)=1 (x=0 is last point)
    D /= np.clip(D[-1], 1e-300, None)

    # Interpolate onto requested z
    # z_grid decreases with x, so reverse for stable interpolation
    z_rev = z_grid[::-1]
    D_rev = D[::-1]
    D_out = np.interp(z_array, z_rev, D_rev)

    return D_out

def fsigma8_DKRD2_ODE(
    z_array,
    v_model: Optional[Callable] = None,
    T_model: Optional[Callable] = None,
    sigma8_0: float = Core_SIGMA8_PLANCK_2018,
    Omega_m0: float | None = None,
    Omega_L_value: float | None = None,
    a_min: float = 1e-3,
    n_grid: int = 6000,
    use_mu_squared: bool = True,
):
    """
    fσ8(z) computed from ODE-based DK growth:

        f(z) = d ln D / d ln a
        fσ8(z) = f(z) * sigma8_0 * D(z)

    D(z) is normalized to D(0)=1 by construction.
    """
    z_array = np.asarray(z_array, dtype=float)

    D = growth_factor_DKRD2_ODE(
        z_array,
        v_model=v_model,
        T_model=T_model,
        Omega_m0=Omega_m0,
        Omega_L_value=Omega_L_value,
        a_min=a_min,
        n_grid=n_grid,
        use_mu_squared=use_mu_squared,
    )

    a = 1.0 / (1.0 + z_array)
    ln_a = np.log(np.clip(a, 1e-300, None))

    idx = np.argsort(ln_a)
    ln_a_sorted = ln_a[idx]
    D_sorted = D[idx]

    lnD_sorted = np.log(np.clip(D_sorted, 1e-300, None))
    dlnD_dlnA = np.gradient(lnD_sorted, ln_a_sorted)

    f = np.empty_like(dlnD_dlnA)
    f[idx] = dlnD_dlnA

    fs8 = f * (float(sigma8_0) * D)
    return fs8, f, D

def fsigma8_LCDM(
    z_array,
    E_LCDM_func,
    Core_H0: float | None,
    Core_c_km_s: float | None = None,
    Core_Omega_m: float | None = None,
    Core_Omega_L: float | None = None,
    sigma8_0: float | None = None,
):
    """
    Compute fσ8(z) for ΛCDM using:
        f(z) = d ln D / d ln a
        fσ8(z) = f(z) * σ8(z) = f(z) * σ8_0 * D(z),

    where D(z) is the linear growth factor normalized to D(0)=1.

    Defaults:
        - If Core_H0 is None, use module-level Core_H0.
        - If Core_c_km_s is None, use module-level Core_c_km_s (kept for API consistency).
        - If Core_Omega_m is None, use module-level Core_Omega_m.
        - If Core_Omega_L is None, use module-level Core_Omega_L.
        - If sigma8_0 is None, use Core_SIGMA8_PLANCK_2018.
    """
    import numpy as np

    z_array = np.array(z_array, dtype=float)

    # Resolve cosmological parameters from globals if needed
    if Core_H0 is None:
        Core_H0_eff = float(globals()["Core_H0"])
    else:
        Core_H0_eff = float(Core_H0)

    if Core_c_km_s is None:
        Core_c_km_s_eff = float(globals()["Core_c_km_s"])
    else:
        Core_c_km_s_eff = float(Core_c_km_s)
    # (Core_c_km_s_eff is not used in the current implementation,
    #  but kept here for possible future extensions and API symmetry.)

    if Core_Omega_m is None:
        Core_Omega_m_eff = float(globals()["Core_Omega_m"])
    else:
        Core_Omega_m_eff = float(Core_Omega_m)

    if Core_Omega_L is None:
        Core_Omega_L_eff = float(globals()["Core_Omega_L"])
    else:
        Core_Omega_L_eff = float(Core_Omega_L)

    if sigma8_0 is None:
        sigma8_0_eff = float(globals()["Core_SIGMA8_PLANCK_2018"])
    else:
        sigma8_0_eff = float(sigma8_0)

    # Compute D(z)
    D = growth_factor_LCDM(
        z_array,
        E_LCDM_func,
        Core_H0_eff,
        Core_Omega_m=Core_Omega_m_eff,
        Core_Omega_L=Core_Omega_L_eff,
    )

    # Compute f(z) numerically via derivative in ln a
    a = 1.0 / (1.0 + z_array)
    ln_a = np.log(a)

    idx = np.argsort(ln_a)
    ln_a_sorted = ln_a[idx]
    D_sorted = D[idx]

    lnD_sorted = np.log(np.clip(D_sorted, 1e-30, None))
    dlnD_dlnA = np.gradient(lnD_sorted, ln_a_sorted)

    f = np.empty_like(dlnD_dlnA)
    f[idx] = dlnD_dlnA

    fs8 = f * (sigma8_0_eff * D)

    return fs8, f, D

def fsigma8_DKRD2(
    z_array,
    v_model,
    T_model,
    sigma8_0: float | None = None,
    Omega_m_DK: float | None = None,
    Omega_L_DK: float | None = None,
):
    """
    Compute fσ8(z) in the DK-RD2 background.

    We compute:
        D_DK(z) from growth_factor_DKRD2_ODE (DK background H(z))
        f(z) = d ln D / d ln a
        fσ8(z) = f(z) * σ8_0 * D(z)

    Defaults:
        - sigma8_0   → Core_SIGMA8_PLANCK_2018
        - Omega_m_DK → Core_Omega_m_fixed_for_DK
        - Omega_L_DK → 0.0  (flat DK-RD2 background by construction)

    If you want a different normalization or background, pass the values explicitly.
    """
    import numpy as np

    z_array = np.array(z_array, dtype=float)

    # Resolve defaults from module-level constants
    if sigma8_0 is None:
        sigma8_0_eff = float(globals()["Core_SIGMA8_PLANCK_2018"])
    else:
        sigma8_0_eff = float(sigma8_0)

    if Omega_m_DK is None:
        Omega_m_DK_eff = float(globals()["Core_Omega_m_fixed_for_DK"])
    else:
        Omega_m_DK_eff = float(Omega_m_DK)

    if Omega_L_DK is None:
        Omega_L_DK_eff = 0.0
    else:
        Omega_L_DK_eff = float(Omega_L_DK)

    # Growth factor on DK-RD2 background
    D = growth_factor_DKRD2_ODE(
        z_array,
        v_model=v_model,
        T_model=T_model,
        Omega_m0=Omega_m_DK_eff,
        Omega_L_value=Omega_L_DK_eff,
    )

    # f(z) = d ln D / d ln a
    a = 1.0 / (1.0 + z_array)
    ln_a = np.log(a)

    idx = np.argsort(ln_a)
    ln_a_sorted = ln_a[idx]
    D_sorted = D[idx]

    lnD_sorted = np.log(np.clip(D_sorted, 1e-30, None))
    dlnD_dlnA = np.gradient(lnD_sorted, ln_a_sorted)

    f = np.empty_like(dlnD_dlnA)
    f[idx] = dlnD_dlnA

    fs8 = f * (sigma8_0_eff * D)

    return fs8, f, D

def growth_factor_LCDM(
    z_array,
    E_LCDM_func,
    Core_H0: float | None,
    Core_Omega_m: float | None = None,
    Core_Omega_L: float | None = None,
    a_min: float = 1e-4,
    n_grid: int = 5000,
):
    """
    Linear growth factor D(z) for ΛCDM, normalized to D(0) = 1.

    Implementation:
        D(a) ∝ H(a) ∫_0^a da' / (a'^3 H(a')^3)

    Defaults:
        - If Core_H0 is None, use module-level Core_H0.
        - If Core_Omega_m is None, use module-level Core_Omega_m.
        - If Core_Omega_L is None, use module-level Core_Omega_L.

    Notes:
        - This assumes GR with scale-independent growth and no modified Poisson term.
    """

    z_array = np.array(z_array, dtype=float)

    # Resolve cosmological parameters from globals if needed
    if Core_H0 is None:
        Core_H0_eff = float(globals()["Core_H0"])
    else:
        Core_H0_eff = float(Core_H0)

    if Core_Omega_m is None:
        Core_Omega_m_eff = float(globals()["Core_Omega_m"])
    else:
        Core_Omega_m_eff = float(Core_Omega_m)

    if Core_Omega_L is None:
        Core_Omega_L_eff = float(globals()["Core_Omega_L"])
    else:
        Core_Omega_L_eff = float(Core_Omega_L)

    # Build an integration grid in scale factor 'a'
    a_grid = np.linspace(a_min, 1.0, int(n_grid))
    z_grid = (1.0 / a_grid) - 1.0

    # Dimensionless expansion history E(z) = H(z)/Core_H0 (provided externally)
    E_grid = np.array([E_LCDM_func(z) for z in z_grid], dtype=float)

    # H(a) = Core_H0 * E(z(a))
    H_grid = Core_H0_eff * E_grid

    # Integral I(a) = ∫ da' / (a'^3 H(a')^3)
    integrand = 1.0 / (a_grid**3 * H_grid**3)

    integral = np.zeros_like(a_grid)
    for i in range(1, len(a_grid)):
        da = a_grid[i] - a_grid[i - 1]
        integral[i] = integral[i - 1] + 0.5 * da * (integrand[i] + integrand[i - 1])

    # Growth factor (unnormalized)
    D_grid = (5.0 * Core_Omega_m_eff * H_grid * integral) / 2.0

    # Normalize to D(a = 1) = 1
    D_grid /= D_grid[-1]

    # Interpolate D(z) on requested z_array
    z_rev = z_grid[::-1]
    D_rev = D_grid[::-1]
    D_out = np.interp(z_array, z_rev, D_rev)

    return D_out

# ============================================================
# SPHEREx utilities (Level2 cubes) for DK-RD2 figures
# ============================================================

def spherex_find_level2_fits(
    folder: str,
    pattern: str = "level2_*.fits",
    *,
    exclude_simulated: bool = True
) -> list[str]:
    """
    Return sorted list of SPHEREx Level2 FITS files inside folder.
    If exclude_simulated=True, filters out any file whose name contains 'simu'.
    """
    import glob
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if exclude_simulated:
        files = [f for f in files if "simu" not in os.path.basename(f).lower()]
    return files

def spherex_read_cube(fits_path: str):
    """
    Read a SPHEREx Level2 FITS cube.

    Returns:
        cube    : ndarray shaped (nz, ny, nx) (spectral axis first)
        wave_um : ndarray (nz,) wavelengths in microns if recoverable else None
        hdr0    : primary header
        meta    : dict with info (hdu index, shape, wave_method)
    """
    from astropy.io import fits

    with fits.open(fits_path, memmap=True) as hdul:
        hdr0 = hdul[0].header
        hdu_idx = None
        data = None
        hdr = None

        for i, h in enumerate(hdul):
            if getattr(h, "data", None) is None:
                continue
            arr = h.data
            if isinstance(arr, np.ndarray) and arr.ndim == 3:
                data = arr
                hdu_idx = i
                hdr = h.header
                break

        if data is None:
            raise ValueError(f"No 3D cube found in: {fits_path}")

        data = np.asarray(data, dtype=np.float32)
        shape = data.shape

        wave_um = None
        wave_method = "none"

        # --- Try spectral WCS on axis 3 ---
        try:
            crval = hdr.get("CRVAL3", None)
            cdelt = hdr.get("CDELT3", None)
            crpix = hdr.get("CRPIX3", 1.0)
            cunit = str(hdr.get("CUNIT3", "")).lower()
            ctype = str(hdr.get("CTYPE3", "")).lower()

            if (crval is not None) and (cdelt is not None):
                # We don't know which physical axis maps to FITS axis3 in raw ndarray,
                # but commonly spectral is stored as last axis in numpy view.
                N = shape[-1]
                pix = np.arange(1, N + 1, dtype=np.float64)
                axis = float(crval) + (pix - float(crpix)) * float(cdelt)

                med = float(np.nanmedian(axis))
                if "m" == cunit or "meter" in cunit:
                    wave_um = axis * 1e6
                    wave_method = "wcs_m"
                elif "um" in cunit or "micron" in cunit:
                    wave_um = axis
                    wave_method = "wcs_um"
                elif "nm" in cunit:
                    wave_um = axis * 1e-3
                    wave_method = "wcs_nm"
                else:
                    # best-effort guess from magnitude
                    if 0.1 < med < 50.0:
                        wave_um = axis
                        wave_method = "wcs_assume_um"
                    elif 1e-8 < med < 1e-3:
                        wave_um = axis * 1e6
                        wave_method = "wcs_assume_m"
        except Exception:
            pass

        # --- Normalize cube to (nz, ny, nx): assume last axis is spectral ---
        cube = np.moveaxis(data, -1, 0)  # (nz, ny, nx)
        if (wave_um is not None) and (wave_um.size != cube.shape[0]):
            # mismatch -> discard wavelength axis to avoid wrong band selection
            wave_um = None
            wave_method = wave_method + "_size_mismatch_drop"

        meta = {
            "fits_path": fits_path,
            "hdu_index": hdu_idx,
            "orig_shape": tuple(shape),
            "cube_shape": tuple(cube.shape),
            "wave_method": wave_method,
        }
        return cube, wave_um, hdr0, meta

def spherex_band_average(
    cube_nz_ny_nx: np.ndarray,
    wave_um: np.ndarray | None,
    um_min: float,
    um_max: float,
    *,
    robust: bool = True
) -> np.ndarray:
    """
    Average intensity map across a wavelength band.
    If wave_um is None -> uses full cube average (band = all channels).
    Returns 2D map (ny, nx).
    """
    cube = np.asarray(cube_nz_ny_nx, dtype=np.float64)
    if cube.ndim != 3:
        raise ValueError("cube must be 3D (nz, ny, nx)")

    if wave_um is None:
        band = cube
    else:
        w = np.asarray(wave_um, dtype=np.float64)
        m = np.isfinite(w) & (w >= float(um_min)) & (w <= float(um_max))
        if not np.any(m):
            raise ValueError(f"No wavelengths within band [{um_min}, {um_max}] um")
        band = cube[m, :, :]

    return np.nanmedian(band, axis=0) if robust else np.nanmean(band, axis=0)

def _planck_B_lambda_ratio_T_lookup(
    lambda1_um: float,
    lambda2_um: float,
    Tmin: float = 2.7,
    Tmax: float = 200.0,
    ngrid: int = 4000,
):
    """
    Lookup table for the ratio B(λ1, T) / B(λ2, T) using Planck's law in λ.

    This routine requires the global constants:
      - Core_h_Planck
      - Core_c_light
      - Core_k_B
    """

    h = float(Core_h_Planck) # Planck constant [J·s]
    c = float(Core_c_light)
    k = float(Core_k_B) # Boltzmann constant [J/K]

    lam1 = float(lambda1_um) * 1e-6  # µm → m
    lam2 = float(lambda2_um) * 1e-6  # µm → m

    T = np.linspace(float(Tmin), float(Tmax), int(ngrid), dtype=np.float64)

    def B(lam, T_arr):
        x = (h * c) / (lam * k * T_arr)
        x = np.clip(x, 1e-9, 700.0)
        return (2.0 * h * c**2) / (lam**5) / np.expm1(x)

    r = B(lam1, T) / B(lam2, T)
    return T, r

def spherex_color_temperature_map(
    I1: np.ndarray,
    I2: np.ndarray,
    lambda1_um: float,
    lambda2_um: float,
    *,
    Tmin: float = 2.7,
    Tmax: float = 200.0
) -> np.ndarray:
    """
    Compute a color-temperature proxy map from two band-averaged intensity maps.
    Returns T_map in Kelvin (NaN where invalid).
    """
    I1 = np.asarray(I1, dtype=np.float64)
    I2 = np.asarray(I2, dtype=np.float64)
    if I1.shape != I2.shape:
        raise ValueError("I1 and I2 must have same shape")

    with np.errstate(divide="ignore", invalid="ignore"):
        R = I1 / I2

    T_grid, R_grid = _planck_B_lambda_ratio_T_lookup(lambda1_um, lambda2_um, Tmin=Tmin, Tmax=Tmax)

    idx = np.argsort(R_grid)
    R_sorted = R_grid[idx]
    T_sorted = T_grid[idx]

    T_map = np.full(I1.shape, np.nan, dtype=np.float64)
    m = np.isfinite(R) & (R > 0.0) & np.isfinite(I1) & np.isfinite(I2) & (I2 > 0.0)

    if np.any(m):
        Rm = np.clip(R[m], float(np.nanmin(R_sorted)), float(np.nanmax(R_sorted)))
        T_map[m] = np.interp(Rm, R_sorted, T_sorted)

    return T_map

def dkrd2_gab_over_g0_for_photons(T_map: np.ndarray):

    T = np.asarray(T_map, dtype=np.float64)
    v = float(Core_c_light)

    with np.errstate(divide="ignore", invalid="ignore"):
        return Gab(T, v) / float(Core_G0)

if __name__ == "__main__":

    print("=" * 80)
    print("DK_RD2_Core.py — Quick sanity checks (CLASS-free)")
    print("-" * 80)

    # Neutral histories if you don't provide custom v(z), T(z)
    V_MODEL = None
    T_MODEL = None

    z_array = np.array([0.01, 0.10, 0.50, 1.00], dtype=float)

    # Make the density parameters explicit for both branches
    Omega_m_LCDM = float(Core_Omega_m)
    Omega_L_LCDM = float(Core_Omega_L)

    # DK-RD2 uses its fixed normalization (not a fitted parameter)
    Omega_m_DK = float(Core_Omega_m_fixed_for_DK)

    # --- Distance modulus μ(z) ---
    mu_LCDM, _ = luminosity_distance_and_mu(
        z_array,
        E_LCDM,
        Core_H0,
        Core_c_km_s=Core_c_km_s,
        Core_Omega_m=Omega_m_LCDM,
        Core_Omega_L=Omega_L_LCDM,
    )

    mu_DK, _ = luminosity_distance_and_mu(
        z_array,
        E_Relativistic,
        Core_H0,
        Core_c_km_s=Core_c_km_s,
        Core_Omega_m=Omega_m_DK,   # explicit DK-RD2 Ωm
        Omega_L_value=None,
        v_model=V_MODEL,
        T_model=T_MODEL,
    )

    print("Distance modulus μ(z):")
    for zi, m1, m2 in zip(z_array, mu_LCDM, mu_DK):
        print(f"  z={zi:>4.2f}  μ_LCDM={m1:8.4f}   μ_DK-RD2={m2:8.4f}")
    print("-" * 80)

    # --- H(z) ---
    Hz_LCDM = H_LCDM(z_array, Core_H0, Omega_m_LCDM, Omega_L_LCDM)
    Hz_DK = H_Relativistic(
        z_array,
        Core_H0,
        Core_Omega_m=Omega_m_DK,   # explicit DK-RD2 Ωm
        Omega_L_value=None,
        v_model=V_MODEL,
        T_model=T_MODEL,
    )

    print("H(z) [km/s/Mpc]:")
    for zi, h1, h2 in zip(z_array, Hz_LCDM, Hz_DK):
        print(f"  z={zi:>4.2f}  H_LCDM={h1:9.3f}   H_DK-RD2={h2:9.3f}")
    print("-" * 80)

    # --- Comoving distance χ(z) ---
    chi_LCDM = comoving_distance(
        z_array,
        E_LCDM,
        Core_H0,
        Core_c_km_s=Core_c_km_s,
        Core_Omega_m=Omega_m_LCDM,
        Core_Omega_L=Omega_L_LCDM,
    )

    chi_DK = comoving_distance(
        z_array,
        E_Relativistic,
        Core_H0,
        Core_c_km_s=Core_c_km_s,
        Core_Omega_m=Omega_m_DK,   # explicit DK-RD2 Ωm
        Omega_L_value=None,
        v_model=V_MODEL,
        T_model=T_MODEL,
    )

    print("Comoving distance χ(z) [Mpc]:")
    for zi, c1, c2 in zip(z_array, chi_LCDM, chi_DK):
        print(f"  z={zi:>4.2f}  χ_LCDM={c1:10.3f}   χ_DK-RD2={c2:10.3f}")
    print("-" * 80)

    # --- Lensing (example with LCDM geometry; shows API is wired) ---
    z_l, z_s = 0.50, 2.00
    sigma_v_kms = 220.0
    th_rad, th_arc = einstein_radius_SIS(
        z_l,
        z_s,
        sigma_v_kms * 1000.0,
        E_LCDM,
        Core_H0,
        Core_Omega_m=Omega_m_LCDM,
        Core_Omega_L=Omega_L_LCDM,
    )
    print("Lensing (SIS) with ΛCDM geometry:")
    print(f"  z_l={z_l:.2f}, z_s={z_s:.2f}, sigma_v={sigma_v_kms:.1f} km/s")
    print(f"  θ_E = {th_rad:.3e} rad  = {th_arc:.3f} arcsec")
    print("-" * 80)

    print("CMB D_ℓ: No Boltzmann backend here. For Planck comparisons,")
    print("provide (ℓ, D_ℓ^ref) to Dl_Relativistic(...) or remap_morph_template(...).")
    print("-" * 80)
    print("Sanity checks finished. Module ready for import.")
    print("=" * 80)
