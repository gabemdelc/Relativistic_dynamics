# coding: utf-8
################################################################################
# DK_RD2_CLASS.py — Option B: Use CLASS for perturbations, force DK-RD2 geometry
################################################################################
# Author:        GabE=mc² (gabemdelc@gmail.com)
# Created:       30/Aug/2025
# License:       MIT
#
# Purpose
# -------
# This module provides a CLASS-backed path to obtain a reference CMB TT spectrum
# (amplitudes, phases from a Boltzmann solver) and then morph it so that its
# geometry (acoustic scale and Silk damping) matches the DK-RD2 background
# computed in DK_RD2_Core.py. This is a pragmatic bridge until a full
# perturbative implementation of DK-RD2 exists inside a Boltzmann code.
#
# What it does (Option B)
# -----------------------
# 1) Run CLASS (via `classy`) with a conventional flat ΛCDM background to get
#    the lensed TT spectrum C_ell^TT (or D_ell).
# 2) Using DK_RD2_Core, compute the acoustic distance quantities for both
#    backgrounds: D_A(z_*), r_s(z_*), l_D (damping scale proxy).
# 3) Morph the reference D_ell to DK-RD2 with:
#       (a) horizontal shift by theta_s = r_s / D_A  (peak positions)
#       (b) multiplicative Silk envelope adjustment (tail suppression)
#    No free fudge factors or amplitude rescaling are introduced here.
#
# Requirements
# ------------
# - CLASS + Python wrapper `classy` installed on your machine.
#   Typical install (system-dependent):
#       git clone https://github.com/lesgourg/class_public
#       cd class_public; make; pip install ./python
################################################################################

from __future__ import annotations

import numpy as np
from typing import Tuple

# Import DK core (geometry, distances, morph helpers, evidence)
from DK_RD2_Core import (
    Core_TCMB_K, E_LCDM, E_Relativistic,
    angular_diameter_distance, r_s_integral,
    remap_morph_template,
    z_star_EH98,
    generate_evidence, load_planck_tt, compute_model_metrics
)

# --- Units helper (CMB temperature) ---
_CONV_K2_TO_uK2 = (Core_TCMB_K * 1e6)**2

def _parse_optional_float(x):
    import numpy as np
    if x is None: return None
    if isinstance(x, str) and x.strip().lower() in ('none', ''): return None
    try:
        v = float(x)
        return None if np.isnan(v) else v
    except Exception:
        return None

def _to_Dl_uK2(ell, y, interpret_as: str):
    """
    Convert a spectrum to D_ell in micro-K^2.

    interpret_as ∈ {'Dl_uK2', 'Dl_normTcmb', 'Cl_uK2', 'Cl_K2'}:
      - 'Dl_uK2'     : y already is D_ell in μK² → return as-is.
      - 'Dl_normTcmb': y is D_ell / Tcmb^2      → multiply by (Tcmb*1e6)^2.
      - 'Cl_uK2'     : y is C_ell in μK²        → D_ell = l(l+1) C_ell / 2π.
      - 'Cl_K2'      : y is C_ell in K²         → convert to μK² and to D_ell.
    """
    import numpy as _np
    ell = _np.asarray(ell, dtype=float)
    y   = _np.asarray(y,   dtype=float)

    if interpret_as == 'Dl_uK2':
        return y
    elif interpret_as == 'Dl_normTcmb':
        return y * _CONV_K2_TO_uK2
    elif interpret_as == 'Cl_uK2':
        return (ell*(ell+1)/(2.0*_np.pi)) * y
    elif interpret_as == 'Cl_K2':
        return (ell*(ell+1)/(2.0*_np.pi)) * y * 1e12
    else:
        raise ValueError(f"Unknown interpret_as: {interpret_as}")


def classy_params_dkrd2_massless(**kw):
    kw.update(dict(N_ncdm=0, m_ncdm=None, N_ur=3.046))
    return classy_params_template(**kw)

def classy_params_planck_baseline(m_nu_eV: float = 0.06, **kw):
    kw.update(dict(N_ncdm=1, m_ncdm=str(m_nu_eV), N_ur=2.0328))
    return classy_params_template(**kw)

# ---------------------------
# 0) Lazy import of `classy`
# ---------------------------
def _ensure_classy():
    try:
        from classy import Class  # type: ignore
        return Class
    except Exception as e:
        raise RuntimeError(
            "CLASS (classy) is not available. Please install the CLASS Boltzmann code "
            "and its Python wrapper `classy`. On many systems:\n"
            "  git clone https://github.com/lesgourg/class_public\n"
            "  cd class_public && make && pip install ./python\n"
            f"Underlying import error: {e}"
        )

from typing import Optional, Dict, Any

def classy_params_template(
    *,
    Core_H0: Optional[float] = None,
    Core_Omega_m: Optional[float] = None,
    Core_Omega_b: Optional[float] = None,
    n_s: Optional[float] = None,
    A_s: Optional[float] = None,
    tau_reio: Optional[float] = None,
    N_ur: Optional[float] = None,
    m_ncdm: Optional[str] = None,   # e.g. '0.06' or '0.03,0.03'
    N_ncdm: int = 0,
    lmax: Optional[int] = None,
    T_cmb: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Build a CLASS parameter dictionary using DK_RD2_Core as the single source of truth.

    Defaults are resolved from DK_RD2_Core:
      - H0            ← Hubble_H0 (or H0)
      - Omega_m       ← Omega_m
      - Omega_b       ← Omega_b
      - n_s           ← NS_PLANCK
      - A_s           ← AS_PLANCK
      - tau_reio      ← TAU_REIO_PLANCK
      - N_ur          ← 3.046 (fixed Standard Model; not defined in core)
      - lmax          ← LMAX_PLANCK
      - T_cmb         ← T_fixed / TCMB_K

    Notes
    -----
    • No ad-hoc constants are set here for DK-RD2; this is only a CLASS
      (ΛCDM-style) parameter packer for reference CMB calculations.
    • Massive neutrinos are added only if N_ncdm > 0.
    """
    # --- import the core once here to avoid circulars in other modules ---
    try:
        import DK_RD2_Core as _core
    except Exception as e:
        raise RuntimeError("DK_RD2_Core must be importable before calling classy_params_template.") from e

    # Small helper to fetch a symbol from core with a fallback
    def _get(name: str, default: Optional[float] = None) -> float:
        val = getattr(_core, name, default)
        if val is None:
            raise RuntimeError(f"Missing required constant '{name}' in DK_RD2_Core and no default supplied.")
        return float(val)

    # --------------------------
    # Resolve all default values
    # --------------------------
    Core_H0       = float(Core_H0)       if Core_H0       is not None else _get("Hubble_H0", getattr(_core, "H0", None))
    Core_Omega_m  = float(Core_Omega_m)  if Core_Omega_m  is not None else _get("Omega_m", None)
    Core_Omega_b  = float(Core_Omega_b)  if Core_Omega_b  is not None else _get("Omega_b", None)
    n_s      = float(n_s)      if n_s      is not None else _get("NS_PLANCK", 0.965)
    A_s      = float(A_s)      if A_s      is not None else _get("AS_PLANCK", 2.1e-9)
    tau_reio = float(tau_reio) if tau_reio is not None else _get("TAU_REIO_PLANCK", 0.054)
    lmax     = int(lmax)       if lmax     is not None else int(_get("LMAX_PLANCK", 2500))
    T_cmb    = float(T_cmb)    if T_cmb    is not None else float(getattr(_core, "T_fixed", getattr(_core, "TCMB_K", 2.7255)))
    N_ur     = float(N_ur)     if N_ur     is not None else 3.046  # Standard Model effective massless ν

    # Basic derived pieces
    h = Core_H0 / 100.0
    Omega_cdm = Core_Omega_m - Core_Omega_b
    if Omega_cdm <= 0:
        raise ValueError("Omega_cdm = Omega_m - Omega_b must be > 0. Check your Omega_m / Omega_b sources.")

    # --------------------------
    # Compose CLASS dictionary
    # --------------------------
    pars: Dict[str, Any] = {
        # Background (CLASS expects physical densities ω = Ω h²)
        "h": h,
        "omega_b": Core_Omega_b * h**2,
        "omega_cdm": Omega_cdm * h**2,
        "T_cmb": T_cmb,
        "Omega_k": 0.0,

        # Reionization & primordial spectrum
        "tau_reio": tau_reio,
        "A_s": A_s,
        "n_s": n_s,

        # Massless neutrinos
        "N_ur": N_ur,

        # Outputs
        "output": "tCl lCl",
        "l_max_scalars": lmax,
        "lensing": "yes",
    }

    # Massive neutrinos only if requested
    if int(N_ncdm) > 0:
        pars["N_ncdm"] = int(N_ncdm)
        # CLASS wants a comma-separated string for multiple species
        pars["m_ncdm"] = str(m_ncdm if m_ncdm is not None else "0.06")
        pars["T_ncdm"] = 0.71611  # standard temperature ratio (T_ν/T_γ)

    return pars


# ------------------------------------------------------------
# 2) Run CLASS and return (ell_ref, D_ell^TT) in μK^2 units
# ------------------------------------------------------------
def class_tt_spectrum(params: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run CLASS and return (ell>=2, D_ell^TT [μK^2]) deterministically:
      CLASS -> C_ell [μK^2]  ==>  D_ell [μK^2] via l(l+1)/2π.
    """
    import numpy as np

    # Safe local copy and enforce useful defaults
    p = dict(params)
    lmax = int(p.get("l_max_scalars", p.get("lmax", 2500)))
    p["l_max_scalars"] = lmax
    # ❌ REMOVE this line (it causes your crash):
    # p["l_max_lensing"] = lmax
    # ensure TT is produced; lensing=yes will return lensed Cls
    p.setdefault("output", "tCl lCl pCl")
    p.setdefault("lensing", "yes")

    Class = _ensure_classy()
    cosmo = Class()
    cosmo.set(p)
    cosmo.compute()

    # CLASS returns C_ell in μK^2; convert to D_ell in μK^2
    cl = cosmo.lensed_cl(lmax)
    ell_all = np.arange(cl["tt"].size, dtype=float)
    m = ell_all >= 2
    ell = ell_all[m]
    Cl_TT_uK2 = np.asarray(cl["tt"][m], dtype=float)
    Dl_TT_uK2 = (ell * (ell + 1) / (2.0 * np.pi)) * Cl_TT_uK2

    cosmo.struct_cleanup(); cosmo.empty()
    return ell, Dl_TT_uK2

# --------------------------------------------------------------------------
# 3) Morph the CLASS template (ΛCDM) to DK-RD2 geometry (no amplitude fit)
# --------------------------------------------------------------------------
def morph_class_to_DKRD2(
    ell_ref: np.ndarray,
    Dl_ref_uK2: np.ndarray,
    class_params: Dict,
    dk_params: Dict,
    *,
    z_star: float | None = None,
    silk_morph: bool = True,
    # NEW: tell us what the morph outputs so we normalize to D_l [μK²]
    remap_output_kind: str = "Dl_normTcmb",   # {'Dl_normTcmb','Dl_uK2','Cl_K2','Cl_uK2'}
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Return (ell, D_ell^TT_DK [μK²], meta) by morphing the CLASS reference curve."""

    # Reference background (from CLASS params)
    h_ref = float(class_params.get('h', 0.0)) or float(class_params.get('H0', 0.0))/100.0
    if h_ref == 0.0:
        raise ValueError("class_params must provide 'h' or 'H0'.")
    omega_b  = float(class_params.get('omega_b', 0.0))
    omega_cdm = float(class_params.get('omega_cdm', 0.0))
    if omega_b == 0.0 or omega_cdm == 0.0:
        raise ValueError("class_params must include 'omega_b' and 'omega_cdm'.")

    Omega_b_ref = omega_b / (h_ref**2)
    Omega_cdm_ref = omega_cdm / (h_ref**2)
    Omega_m_ref = Omega_b_ref + Omega_cdm_ref
    Omega_L_ref = 1.0 - Omega_m_ref  # flat

    # DK background
    H0_dk       = float(dk_params.get('H0', dk_params.get('Core_H0')))
    if not np.isfinite(H0_dk) or H0_dk <= 0:
        raise ValueError("dk_params must provide 'H0' (or 'Core_H0') as a positive number.")
    Omega_m_dk = _parse_optional_float(dk_params.get('Omega_m'))
    Omega_b_dk  = float(dk_params.get('Omega_b', dk_params.get('Core_Omega_b')))
    if not np.isfinite(Omega_b_dk) or Omega_b_dk < 0:
        raise ValueError("dk_params must provide 'Omega_b' (or 'Core_Omega_b') as a non-negative number.")
    v_model     = dk_params.get('v_model', None)
    T_model     = dk_params.get('T_model', None)

    # Effective baryon density for microphysics (use reference Ω_b h^2)
    Ob_h2 = omega_b  # equals Ω_b_ref * h_ref^2

    if silk_morph:
        # The remap returns an amplitude array in some convention (declare it)
        Dl_dk_raw, meta = remap_morph_template(
            ell=ell_ref, ell_ref=ell_ref, Dl_ref=Dl_ref_uK2,
            Core_H0=H0_dk,
            Core_Omega_m=Omega_m_ref,   # for EH98; Ob_h2 carries physical baryons
            Core_Omega_L=Omega_L_ref,
            Ob_h2=Ob_h2,
            z_star=z_star
        )
        # Normalize to D_l [μK²]
        Dl_dk_uK2 = _to_Dl_uK2(ell_ref, Dl_dk_raw, interpret_as=remap_output_kind)
        meta = dict(meta, silk=True)
    else:
        # Geometry-only fallback
        if z_star is None:
            z_star = z_star_EH98(Ob_h2, Omega_m_ref * h_ref**2)

        DA_ref = angular_diameter_distance(z_star, E_LCDM, Core_H0=h_ref*100.0,
                                           Core_Omega_m=Omega_m_ref, Core_Omega_L=Omega_L_ref)
        rs_ref = r_s_integral(z_star, E_LCDM, Core_H0=h_ref*100.0, Ob_h2=Ob_h2,
                              Core_Omega_m=Omega_m_ref, Core_Omega_L=Omega_L_ref)

        DA_dk = angular_diameter_distance(z_star, E_Relativistic, Core_H0=H0_dk,
                                          Core_Omega_m=Omega_m_dk, Omega_L_value=None,
                                          v_model=v_model, T_model=T_model)
        rs_dk = r_s_integral(z_star, E_Relativistic, Core_H0=H0_dk, Ob_h2=Ob_h2,
                             Core_Omega_m=Omega_m_dk, Omega_L_value=None)

        ellA_ref = np.pi * (DA_ref / rs_ref)
        ellA_dk  = np.pi * (DA_dk  / rs_dk)
        alpha = float(ellA_ref / ellA_dk)
        lprime = np.clip(ell_ref * alpha, ell_ref.min(), ell_ref.max())
        Dl_dk_uK2 = np.interp(lprime, ell_ref, Dl_ref_uK2, left=0.0, right=0.0)

        meta = dict(
            z_star=float(z_star),
            DA_LCDM=float(DA_ref), DA_DK=float(DA_dk),
            rs_LCDM=float(rs_ref), rs_DK=float(rs_dk),
            theta_ratio=float(ellA_ref/ellA_dk),
            silk=False
        )

    return np.asarray(ell_ref, float), np.asarray(Dl_dk_uK2, float), meta

# -----------------------------------------------------------------------
# 4) One-shot helper: CLASS → morph → (ell, D_ell^DK, meta, (ell_ref,Dl))
# -----------------------------------------------------------------------
def cmb_tt_DKRD2_via_CLASS(
    class_params: Dict,
    dk_params: Dict,
    *,
    z_star: float | None = None,
    silk_morph: bool = True,
    remap_output_kind: str = "Dl_normTcmb",   # {'Dl_normTcmb','Dl_uK2','Cl_K2','Cl_uK2'}
) -> Tuple[np.ndarray, np.ndarray, Dict, Tuple[np.ndarray, np.ndarray]]:
    """CLASS template → DK-RD2 morph; returns DK curve and the CLASS reference, both as D_l [μK²]."""
    ell_ref, Dl_ref_uK2 = class_tt_spectrum(class_params)
    ell, Dl_dk_uK2, meta = morph_class_to_DKRD2(
        ell_ref, Dl_ref_uK2, class_params, dk_params,
        z_star=z_star, silk_morph=silk_morph, remap_output_kind=remap_output_kind
    )
    return ell, Dl_dk_uK2, meta, (ell_ref, Dl_ref_uK2)

# ------------------------------------------------------
# 5) χ² against Planck (or other) TT bandpower catalog
# ------------------------------------------------------
def chi2_against_planck(
    ell: np.ndarray,
    Dl_model: np.ndarray,
    planck_tt_path: str,
    *,
    k_params: int = 6,
    return_table: bool = False
):
    """Compute χ² between a model D_ell(ell) and a Planck-like TT table (both in μK²)."""
    ell_obs, Dl_obs, Dl_err = load_planck_tt(planck_tt_path)
    Dl_on_obs = np.interp(ell_obs, ell, Dl_model, left=0.0, right=0.0)

    # quick amplitude sanity to catch unit regressions
    if float(np.nanmax(Dl_on_obs)) < 1e3:
        raise ValueError("chi2_against_planck: model D_ell seems too small; check units.")

    var = np.maximum(Dl_err, 1e-12)**2
    chi2 = float(np.sum((Dl_on_obs - Dl_obs)**2 / var))
    n = int(ell_obs.size)
    aic, bic = compute_model_metrics(chi2, k=k_params, n=n)

    if return_table:
        import pandas as pd
        df = pd.DataFrame({
            'ell': ell_obs,
            'Dl_obs': Dl_obs,
            'Dl_err': Dl_err,
            'Dl_model': Dl_on_obs,
            'residual': Dl_on_obs - Dl_obs
        })
        return chi2, n, aic, bic, df
    else:
        return chi2, n, aic, bic

# ------------------------------------------------------
# 6) Save helpers (optional)
# ------------------------------------------------------
def save_curve_csv(ell: np.ndarray, Dl: np.ndarray, tag: str = "cmb_tt_dk") -> str:
    import pandas as pd
    path = generate_evidence("table", tag)
    pd.DataFrame({'ell': np.asarray(ell, float), 'Dl': np.asarray(Dl, float)}).to_csv(path, index=False)
    return path

def save_meta_json(meta: Dict, tag: str = "cmb_tt_meta") -> str:
    import json
    path = generate_evidence("json", tag)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return path

__all__ = [
    "classy_params_template",
    "class_tt_spectrum",
    "morph_class_to_DKRD2",
    "cmb_tt_DKRD2_via_CLASS",
    "chi2_against_planck",
    "save_curve_csv",
    "save_meta_json",
]
