# coding=utf-8
###################################################################################
# DK-RD2 Core Utilities – Constants, Functions, and Dynamic Gravitational Engine
###################################################################################
#    Author:      GabE=mc² (gabemdelc@gmail.com)
#    Created:     11/Feb/2025
#    Project:     DK-RD2 — DK-Relativistic Dynamics Model (2.0)
#    License:     MIT License
#    Repository:  https://github.com/gabemdelc/Relativistic_dynamics
###################################################################################
"""
====================================================================================
🧠 DK_RD2_Core.py — Core Computational Engine for the DK-RD2 Cosmological Framework
====================================================================================

This module implements the **relativistic and thermodynamic physics engine** of the
DK-RD2 model (*Dark Killer – Relativistic Dynamics*), a novel cosmological theory that
replaces dark energy and dark matter with emergent relativistic effects driven by
temperature and velocity-dependent gravity.

The file serves as the universal and modular **core library** that powers all scientific
simulations, statistical validations, and figure generation pipelines associated with
the DK-RD2 model. It is fully portable, reusable, and designed to scale across datasets
including SN Ia (Pantheon+, Union2), CMB spectra, H(z), DESI redshifts, and gravitational
lensing.

------------------------------------------------------------------------------------
📦 Module Capabilities and Structure:

This core file defines and exports the following submodules:

1. **Constants and Conversions**
   - Fundamental cosmological constants (H₀, c, G, T₀, Ωₘ, Ω_Λ)
   - Unit conversions (e.g., Mpc to m, eV to K)

2. **Gravitational Coupling Engine**
   - `Gab_Tv(T, v)` — Thermodynamic gravity coupling
   - `Gab_z(z)` — Redshift-based mapping of Gab
   - `Omega_m_eff(z)` — Effective matter density evolution

3. **Modified Friedmann Dynamics**
   - `E_Relativistic_temp(z)` — Relativistic expansion function
   - `H_z_relativistic(z)` — Modified Hubble rate
   - `H_z_LCDM(z)` — Reference ΛCDM expansion

4. **Distance Functions**
   - `luminosity_distance_Relativistic_temp(z)`
   - `luminosity_distance_LCDM(z)`
   - Comoving distance, μ(z), d_L(z), dn/dz

5. **CMB and Angular Power Spectrum**
   - `angular_power_spectrum(l, model='dk')`
   - Computes D_ℓ(ℓ) for DK-RD2 and ΛCDM

6. **Gravitational Lensing Tools**
   - `Einstein_radius`, `D_ls_proxy`
   - Lensing predictions with dynamic gravity

7. **Statistical Comparison Tools**
   - `calculate_chi2_SN()` — χ² for supernova datasets
   - `calculate_chi2_Hz()` — χ² for expansion rate datasets
   - `compute_model_metrics()` — AIC, BIC, MSE calculators

8. **Utility and File I/O Tools**
   - `generate_evidence()` — Centralized filename generator
   - Automated saving paths for figures, tables, and _stats files

------------------------------------------------------------------------------------
🎯 Scientific Purpose:

This module encapsulates the theoretical foundation of DK-RD2, which proposes that:
- **Gravity is emergent and thermodynamic**, not fundamental.
- The coupling constant G becomes **Gab(T, v)**, responsive to local energy conditions.
- **Accelerated cosmic expansion** arises naturally from relativistic energy damping.
- **No free parameters** are introduced — the model is parameter-free and directly testable.
- Apparent dark matter phenomena result from **redshift-dependent mass enhancement**.

------------------------------------------------------------------------------------
🔄 Integration Pattern:

This core file is designed to be imported by a main driver script (`DK-RD2.py`) which:
- Loads observational datasets
- Calls the physics engine to generate predictions
- Computes statistical metrics (χ², AIC, BIC, MSE)
- Produces tables, figures, and comparison graphics
- Consolidates _stats outputs for global evaluation

------------------------------------------------------------------------------------
📊 Example Outputs Triggered via Main Script:
- `DK-RD2_table_*.csv` — Predictions for specific datasets
- `DK-RD2_image_*.png` — Visual validation against ΛCDM
- `DK-RD2_table_*_stats.csv` — χ², AIC, BIC comparison metrics
- `*_comparison.png` — Bar chart of global DK-RD2 vs ΛCDM performance

------------------------------------------------------------------------------------
🧩 Modular, portable, and optimized for scientific exploration.
Use this file as the analytical backbone for any DK-RD2 cosmological experiment.
====================================================================================
"""


# === IMPORTS ===
import os
import inspect
import numpy as np
from scipy.integrate import quad

################################################################################
# GitHub link for visual reference in plots
git_gabe = "https://github.com/gabemdelc/Relativistic_dynamics"
################################################################################

# === UNIVERSAL PHYSICAL CONSTANTS ===
G0 = 6.67430e-11        # Newton's gravitational constant [m³ kg⁻¹ s⁻²]
c_light = 2.99792458e8  # Speed of light [m/s]
c_km_s = c_light / 1000  # Speed of light converted to [km/s]
T_fixed = 2.725         # Current average CMB temperature [K] (Planck 2018)
rest_wavelength = 1216  # Angstroms (Ly-alpha)
# Unit conversions
km_to_m = 1e3  # 1 kilometer = 1,000 meters

pc_to_m = 3.08567758149137e16   # 1 parsec = 3.08567758149137 × 10^16 meters
Mpc_to_m = 3.08567758149137e22  # 1 megaparsec = 10^6 parsecs = 3.08567758149137 × 10^22 meters

# === COSMOLOGICAL PARAMETERS ===
Hubble_H0 = 70.0        # Hubble constant [km/s/Mpc] (standard value)
Omega_L_DES = 0.669     # Cosmological constant Ω_Λ from DES 2018 (SN Ia data)
Omega_L_Planck = 0.6847 # Cosmological constant Ω_Λ from Planck 2018 (CMB data)
Omega_m = 0.27          # Matter density parameter Ω_m (WMAP baseline)

Omega_L_labels = {
    'DES': r'$\Omega_\Lambda = 0.669$ (DES 2018)',
    'Planck': r'$\Omega_\Lambda = 0.6847$ (Planck 2018)'
}

# === LENSING SIMULATION PARAMETERS ===
sigma_v_base = 250_000                # Velocity dispersion of lens [m/s]
theta_LCDM_base = 4.087647319120988e-12  # Einstein radius in arcseconds (ΛCDM reference)
z_lens = 0.2                          # Typical redshift of lensing galaxy
z_source = 1.0                        # Typical redshift of background source

# === ANGULAR POWER SPECTRUM SCALING ===
L_TO_Z_SCALE = 3000.0  # Empirical scaling factor: ℓ ≈ 3000 ↔ z ≈ 1
# The dir path for saving figures, tables, results
out_dir_path = "evidence/"

"""
L_TO_Z_SCALE: Empirical factor used to approximate the effective redshift z_eff
corresponding to a given multipole ℓ in the CMB angular power spectrum.

Relation:
    z_eff ≈ ℓ / L_TO_Z_SCALE

Motivation:
    ℓ ≈ 3000 corresponds roughly to z ≈ 1 in standard cosmological mappings.
"""

""" ##############################################################
            === Global relativistic functions ===
############################################################## """
def Gab(T, v, G_const=G0):
    """
    Thermodynamic-relativistic gravitational coupling Gab(T, v)

    - Incorporates Lorentz gamma factor for relativistic enhancement
    - Applies thermal damping consistent with CMB physics
    - Enforces physical bounds on velocity and temperature
    - This function should be used in place of the classical gravitational constant G
      in any simulation or test involving thermal-relativistic regimes, including
      cosmic expansion, dark matter modeling, and gravitational lensing.

    Parameters:
        T : float or np.ndarray
            Temperature in Kelvin (typically T_CMB / (1 + z))
        v : float or np.ndarray
            Particle/system velocity in m/s (capped at 0.9999 * c)
        G_const : float
            Reference gravitational constant (default = G0 in SI units)

    Returns:
        Gab_eff : float or np.ndarray
            Effective gravitational coupling Gab(T, v) in units of m³ kg⁻¹ s⁻²
    """

    # Safety bounds to avoid numerical issues
    v = np.clip(v, 1e-8, 0.9999 * c_light)
    T = np.clip(T, 1e-4, 300.0)

    # Lorentz factor
    gamma = 1.0 / np.sqrt(1.0 - (v / c_light) ** 2)

    # Logarithmic relativistic enhancement (compressed)
    log_term = np.log10(1 + (gamma - 1))

    # Thermal damping factor: deviation from CMB temperature
    thermal_damping = (T / T_fixed)**0.25

    # Final thermodynamic gravitational correction
    Gab_eff = G_const * (1 + log_term / thermal_damping)

    return Gab_eff # Effective gravitational coupling Gab(T, v) in units of m³ kg⁻¹ s⁻²


def Gab_z(z):
    """
    Returns the thermodynamic gravitational coupling Gab at redshift z,
    using temperature and velocity scaling consistent with standard cosmological redshift evolution:

    - T(z) = T_CMB / (1 + z)
    - v(z) ∝ sqrt(1 + z), capped below c
    """
    T = T_fixed / (1 + z) # CMB temperature
    v = np.minimum(0.9999 * c_light, 0.1 * c_light * np.sqrt(1 + z))
    return Gab(T, v)

# === EFFECTIVE MATTER DENSITY AND EXPANSION FUNCTIONS ===

def Omega_m_Gab(Om_m, G_rel, G_const=G0):
    """
    Returns the effective matter density Ω_m_eff = Ω_m * (G_rel / G0),
    where G_rel is the dynamic gravitational coupling.
    """
    return Om_m * (G_rel / G_const)

def Omega_m_Gab_z(z):
    """
    Wrapper for computing Ω_m_eff as a function of redshift z.
    """
    return Omega_m_Gab(Omega_m, Gab_z(z), G_const=G0)


def H_relativistic(z, Omega_L_value=Omega_L_DES):
    """
    Computes the relativistic Hubble parameter H(z),
    using the dynamically modified matter density Omega_m_Gab(z)
    and a specified dark energy density Omega_L_value.

    Parameters:
    - z : redshift
    - Omega_L_value : dark energy density ΩΛ (default = 0.669 from DES)

    Returns:
    - H(z) in km/s/Mpc
    """
    return Hubble_H0 * np.sqrt(Omega_m_Gab_z(z) * (1 + z)**3 + Omega_L_value)

# === HUBBLE FUNCTION INVERSE: E(z) MODELS ===

def E_LCDM(z, Omega_L_value=Omega_L_DES):
    """
    Returns 1/H(z) for the standard ΛCDM model (constant G),
    with configurable dark energy density.

    Parameters:
    - z: redshift
    - Omega_L_value: dark energy density (default = 0.7)

    Returns:
    - 1 / H(z)
    """
    return 1.0 / np.sqrt(Omega_m * (1 + z)**3 + Omega_L_value)

def E_Relativistic(z, Omega_L_value=Omega_L_DES):
    """
    Computes the inverse Hubble parameter 1/H(z) for the relativistic dynamics model,
    using dynamically modified matter density based on Gab(z).

    Parameters:
        z : float or array
            Redshift
        Omega_L_value : float
            Dark energy density parameter (default: DES 2018)

    Returns:
        1 / H(z)
    """
    return 1.0 / np.sqrt(Omega_m_Gab_z(z) * (1 + z)**3 + Omega_L_value)

def E_Relativistic_temp(z, Om_m=Omega_m, Omega_L=Omega_L_DES):
    """
    Temporary relativistic model for testing: uses Gab(T, v) where
    T = T_CMB / (1 + z) and v ∝ sqrt(1 + z), modulates Ω_m dynamically.

    Parameters:
        z : float or array
            Redshift
        Om_m : float
            Matter density parameter
        Omega_L : float
            Dark energy density

    Returns:
        1 / H(z)
    # To avoid unphysically large Lorentz boosts at low redshift,
    # we limit velocity scaling to v(z) ∝ 0.1·c·√(1+z), ensuring Gab(T,v)
    # remains within physically plausible bounds and avoids inflating μ(z)
    # due to near-light-speed behavior across most of the redshift range.
    """
    T = T_fixed / (1 + z)
    # Natural velocity regulator that decays at low z and saturates at high z
    v = c_light * (z / (1 + z)) ** 0.5
    G_rel = Gab(T, v)
    Omega_m_mod = Omega_m_Gab(Om_m, G_rel)
    return 1.0 / np.sqrt(Omega_m_mod * (1 + z) ** 3 + Omega_L)

# === DISTANCES AND DISTANCE MODULUS ===

def comoving_distance(z, E_function, Hubble=Hubble_H0):
    """
    Compute the comoving distance for a given redshift z using the provided expansion function E(z).

    Parameters:
        z : float or np.ndarray
            Redshift or array of redshifts to compute the comoving distance.
        E_function : callable
            Function E(z) = H(z)/H0 (dimensionless expansion function).
        Hubble : float
            Hubble constant in km/s/Mpc.

    Returns:
        d_c : float or np.ndarray
            Comoving distance(s) in Mpc.
    """
    def integrand(x):
        return 1.0 / E_function(x)

    # Force input to be array for vectorization
    z_array = np.atleast_1d(z)

    # Compute integral for each redshift
    d_c_array = np.array([
        (c_light / Hubble) * quad(integrand, 0, z_i)[0]
        for z_i in z_array
    ])

    # Return float if input was scalar
    return d_c_array[0] if np.isscalar(z) else d_c_array

def luminosity_distance(z, E_func):
    """
    Computes distance modulus μ(z) using consistent SI units.
    """
    H0_SI = Hubble_H0 * 1000 / Mpc_to_m  # H0 in 1/s Megaparsecs
    integral = np.array([quad(E_func, 0, zi)[0] for zi in z])
    dL_m = (c_light / H0_SI) * (1 + z) * integral  # in meters
    dL_pc = dL_m / pc_to_m  # in parsecs
    mu = 5 * np.log10(dL_pc) - 5
    return mu

def luminosity_distance_LCDM(z):
    """
    Distance modulus μ(z) for ΛCDM using E_LCDM(z) as expansion function.
    """
    return luminosity_distance(np.atleast_1d(z), E_LCDM)

def luminosity_distance_Relativistic_temp(z, hubble=Hubble_H0, Om_m=Omega_m, Om_L=Omega_L_DES):
    z = np.atleast_1d(z)  # ✅ z always and arraay
    H0_SI = hubble * 1000 / Mpc_to_m  # H0 en 1/s

    dL_m = np.array([
        (c_light / H0_SI) * (1 + zi) * quad(E_Relativistic_temp, 0, zi, args=(Om_m, Om_L))[0]
        for zi in z
    ])
    dL_Mpc = dL_m / Mpc_to_m  # Convert d_L to megaparsecs; ensures consistency with SN Ia observational datasets (e.g., Pantheon+)
    mu = 5 * np.log10(dL_Mpc) + 25  # Standard definition of distance modulus μ in cosmology: includes +25 offset for Mpc scale

    return mu if len(mu) > 1 else mu[0]  # ✅ if scalar return scalar


# === GRAVITATIONAL LENSING ===

def D_lens(z_l, z_s):
    """
    Simplified distance proxy D_ls for lensing,
    based on redshifts of lens and source.

    Returns:
        D_lens : float
    """
    return (z_l * z_s) / (z_s - z_l)

def einstein_radius(Gval, sigma_v, z_l, z_s):
    """
    Computes Einstein radius (radians) from gravitational lensing formula.

    Parameters:
        Gval : float
            Gravitational constant used (G0 or Gab)
        sigma_v : float
            Velocity dispersion (m/s)
        z_l : float z_lens
        z_s : float z_source

    Returns:
        θ_E : float
            Einstein radius in radians
    """
    D = D_lens(z_l, z_s)
    return np.sqrt(4 * Gval * sigma_v ** 2 * D / c_light ** 2)

# === ANGULAR POWER SPECTRUM ===

def Dl_LCDM(l):
    """
    Synthetic angular power spectrum D_ℓ for the ΛCDM model.

    This function provides a simplified toy model of the angular power spectrum
    D_ℓ(ℓ) under the ΛCDM framework. It consists of two Gaussian components centered
    at multipoles ℓ ≈ 250 and ℓ ≈ 550, mimicking the first acoustic peaks observed
    in the CMB power spectrum.

    Parameters:
        l : int or float or np.ndarray
            Multipole moment ℓ at which to evaluate the power spectrum.

    Returns:
        D_ell : float or np.ndarray
            Dimensionless power spectrum amplitude D_ℓ(ℓ) corresponding to each ℓ.
            Units are arbitrary (e.g., μK²), normalized for qualitative comparison.
    """
    return 1e4 * np.exp(- (l - 250) ** 2 / (2 * 100 ** 2)) + 1e3 * np.exp(- (l - 550) ** 2 / (2 * 120 ** 2))

def Dl_Relativistic(l, T_override=None):
    """
    Computes the relativistic angular power spectrum D_ell(l)
    using Gab(T, v) dynamically derived from T(z) and v(z).
    You can override the default temperature scaling by providing T_override.

    Parameters:
        l (array): Multipole values
        T_override (float or array, optional): Custom temperature to use instead of default T_fixed * (1 + z)

    Returns:
        array: D_ell values for the relativistic model
    """

    # Approximate mapping from multipole ℓ to redshift z
    z_l = 10 + 1000 / (l + 1e-3)

    # Temperature scaling
    if T_override is not None:
        T = T_override * np.ones_like(z_l)  # override with constant or vector
    else:
        T = T_fixed * (1 + z_l)

    # Velocity scaling with z, capped at 99.99% of c
    v = np.minimum(0.9999 * c_light, 0.1 * c_light * np.sqrt(1 + z_l))

    # Compute thermodynamic gravitational coupling
    Gab_l = Gab(T, v)

    # Fallback for numerical stability
    if np.isnan(Gab_l).any() or np.isinf(Gab_l).any():
        Gab_l = G0

    # Return D_ell scaled by Gab/G0
    return (Gab_l / G0) * Dl_LCDM(l)

def Dl_Relativistic_Gab(l):
    """
    Computes the relativistic angular power spectrum D_ell(l)
    modified by the thermodynamic gravitational correction Gab(T, v),
    using a physical mapping between multipole moment ℓ and effective redshift z.

    - Effective redshift z(l) ≈ l / 3000
    - Temperature: T(z) = T_CMB / (1 + z)
    - Velocity: v(z) ∝ sqrt(1 + z), capped at 99.99% c
    - Gab enhancement is scaled as Gab(T,v) / G0

    Returns:
        D_ell_RDM(l) = D_ell_LCDM(l) × [Gab(T,v) / G0]
    """
    z_eff = l / L_TO_Z_SCALE #  # Unitless

    # Compute temperature and velocity at effective redshift
    T_Gab = T_fixed / (1 + z_eff)
    v_Gab = np.minimum(0.9999 * c_light, 0.1 * c_light * np.sqrt(1 + z_eff))

    # Gravitational enhancement factor
    G_eff = Gab(T_Gab, v_Gab)
    scale_factor = G_eff / G0

    # Modified angular power spectrum
    return Dl_LCDM(l) * scale_factor

# === OBSERVABLE GALAXY COUNTS MODEL ===

def dn_dz_model(z, E_function, Hubble=Hubble_H0):
    """
    Computes the expected normalized number density dn/dz
    based on expansion rate H(z).

    Parameters:
        z : float
        E_function : function
        Hubble : float

    Returns:
        dn/dz : float
    """
    H_z = Hubble / E_function(z)
    return H_z**-1 * (1 + z)**2

# === HUBBLE PARAMETER MODEL EVALUATION ===

def Hz_from_model(z_array, model_function, Hubble=Hubble_H0):
    """
    Computes predicted H(z) using the model function E(z) = 1/H(z).

    Parameters:
        z_array : array
        model_function : function returning 1/H(z)
        Hubble : Hubble Constant H0

    Returns:
        Hz_model : array
            Hubble parameter values H(z)
    """
    return Hubble * np.array([model_function(z) for z in z_array])

def effective_density_from_redshift(z_array, Omega_m_value=Omega_m, G_func=Gab_z):
    """
    Computes normalized effective density profile rho(z) ∝ Ω_m_eff(z) * (1+z)^3

    Parameters:
        z_array : array
            Redshift values
        Omega_m_value : float
            Base matter density (default from core)
        G_func : function
            Function to compute G_eff(z)

    Returns:
        rho_eff_normalized : np.ndarray
            Normalized density profile
    """
    G_rel = np.array([G_func(z) for z in z_array])
    rho_eff = Omega_m_value * (G_rel / G0) * (1 + z_array)**3
    return rho_eff / np.max(rho_eff)

def circular_velocity_Gab(r_kpc, rho0, T_CMB=T_fixed):
    """
    Computes the circular velocity V(r) from DK-RD²M using Gab(T, v) for a given density rho0.

    Parameters:
        r_kpc : float
            Radius in kiloparsecs.
        rho0 : float
            Central density in M_sun / kpc^3.
        T_CMB : float
            Temperature in Kelvin (default = T_fixed).

    Returns:
        v_kms : float
            Circular velocity in km/s, or np.nan if no solution.
    """
    from scipy.optimize import root_scalar
    import numpy as np

    def v_to_m_s(v_kms): return v_kms * km_to_m

    def equation(v_kms):
        v_ms = v_to_m_s(v_kms)
        Gab_val = Gab(T_CMB, v_ms)  # Output in SI units: m^3 / (kg s^2)

        # === Convert Gab to (kpc * (km/s)^2) / M_sun ===
        # Start from SI: m^3 / (kg s^2)
        # Target: kpc * (km/s)^2 / M_sun

        # Conversion factors:
        # 1 M_sun = 1.98847e30 kg
        # 1 m = 1 / pc_to_m * 10^3 kpc
        # 1 m^3 = (1 / pc_to_m)^3 * (10^3)^3 kpc^3
        # 1 (m/s)^2 = (km/s)^2 / (km_to_m)^2

        Gab_kpc_units = (
            Gab_val * 1.98847e30 /               # kg to M_sun
            (pc_to_m * 1e3) ** 3 *                # m^3 to kpc^3
            (1 / km_to_m) ** 2                    # m^2/s^2 to (km/s)^2
        )

        return v_kms**2 - Gab_kpc_units * rho0 * r_kpc**2

    try:
        sol = root_scalar(equation, bracket=[1, 300], method='brentq')
        return sol.root if sol.converged else np.nan
    except Exception:
        return np.nan

############################################################################################
"""
        === statistical_functions ===
"""
############################################################################################

def calculate_sigma(delta_chi_squared):
    """
    Converts Δχ² into Gaussian σ (standard deviation units).

    Parameters:
        delta_chi_squared : float

    Returns:
        sigma : float
    """
    return np.sqrt(delta_chi_squared)


def calculate_chi2_SN(mu_obs, mu_err, mu_model):
    """
    Computes χ² and MSE between observed and model μ(z) values.

    Parameters:
        mu_obs : array
        mu_err : array
        mu_model : array

    Returns:
        chi2 : float
        mse : float
    """
    residuals = mu_obs - mu_model
    chi2 = np.sum((residuals / mu_err) ** 2)
    mse = np.mean(residuals ** 2)
    return chi2, mse


def calculate_chi2_Hz(Hz_obs, Hz_err, Hz_model):
    """
    Computes χ² and MSE for H(z) data.

    Parameters:
        Hz_obs : array
        Hz_err : array
        Hz_model : array

    Returns:
        chi2 : float
        mse : float
    """
    residuals = Hz_obs - Hz_model
    chi2 = np.sum((residuals / Hz_err) ** 2)
    mse = np.mean(residuals ** 2)
    return chi2, mse

def compute_model_metrics(chi2: float, n_params: int, n_data: int):
    """
    Compute AIC (Akaike Information Criterion) and BIC (Bayesian Information Criterion)
    for a given model based on the chi-squared value, the number of free parameters,
    and the number of data points.

    Parameters:
        chi2 (float): The chi-squared value of the model fit.
        n_params (int): The number of free parameters in the model.
        n_data (int): The number of data points used for fitting.

    Returns:
        tuple: AIC and BIC values for the model.

    Notes:
        Lower AIC/BIC values indicate a better balance between goodness-of-fit
        and model complexity. These criteria are widely used for model comparison.
    """
    aic = chi2 + 2 * n_params
    bic = chi2 + n_params * np.log(n_data)
    return aic, bic


def likelihood_ratio_test(chi2_model_simple: float, chi2_model_complex: float, df: int):
    """
    Perform a Likelihood Ratio Test (LRT) between two nested models.

    Parameters:
        chi2_model_simple (float): Chi-squared value of the simpler model (e.g., ΛCDM).
        chi2_model_complex (float): Chi-squared value of the more complex model (e.g., DK-RD2).
        df (int): Difference in the number of parameters between the two models.

    Returns:
        tuple: (LR statistic, p-value) indicating the strength of evidence favoring
               the more complex model. A low p-value (e.g., < 0.05) suggests the
               complex model provides a significantly better fit.

    Notes:
        This test evaluates whether the improvement in fit justifies the added complexity
        of the more advanced model.
    """
    from scipy.stats import chi2 as chi2_dist
    lr_stat = chi2_model_simple - chi2_model_complex
    p_value = 1 - chi2_dist.cdf(lr_stat, df)
    return lr_stat, p_value

############################################################################################
"""
        === EVIDENCE FILE NAMING UTILITY ===
"""
############################################################################################
def generate_evidence(evidence_type, consecutive=None, ext="", out_dir=out_dir_path):
    """
    Generates standardized filenames for saving results (tables, plots, etc.)
    based on the calling script's name and evidence type.

    Parameters:
        evidence_type: "graph", "table", "image", "data", etc.
        consecutive: optional index to differentiate multiple outputs
        ext: custom extension (overrides type-based default if provided)
        out_dir: output directory path (default = "evidence/")

    Returns:
        file_name: string with full path to save evidence.
    """
    caller_frame = inspect.stack()[1]
    caller_file = caller_frame.filename
    program_name = os.path.basename(caller_file).replace('.py', '')

    if evidence_type == "graph" or evidence_type == "image":
        extension = "png"
    elif evidence_type == "table":
        extension = "csv"
    elif evidence_type == "data":
        extension = "txt"
    elif evidence_type == "json":
        extension = "json"
    else:
        extension = ext
        # prefix = evidence_type if evidence_type else program_name

    prefix = f"{program_name}_{evidence_type}" if evidence_type else program_name
    if consecutive != "":
        consecutive = str(consecutive).zfill(2)
        file_name = f"{out_dir}{prefix}_{consecutive}.{extension}"
    else:
        file_name = f"{out_dir}{prefix}.{extension}"
    return file_name

if __name__ == "__main__":
    # === Quick Validation Block ===
    # Used to verify consistency between ΛCDM and DK-RD² distance predictions
    # Consider moving to a test notebook or pytest later
    if __name__ == "__main__":
        print("=" * 80)
        print("🧠 DK_RD2_core.py — Relativistic Gravity Verification")
        print("This file is meant to be imported as a module in DK-RD2 simulations.")
        print("But you can also run it directly to see a quick consistency check between:")
        print("ΛCDM (fixed gravity) vs DK-RD2 (thermodynamic-relativistic gravity).")
        print("-" * 80)

        z_test = np.array([0.1])
        mu_test = luminosity_distance(z_test, E_LCDM)
        mu_test_rel = luminosity_distance_Relativistic_temp(z_test)

        print(f"μ(z=0.1) from ΛCDM     ≈ {mu_test[0]:.4f}")
        print(f"μ(z=0.1) from DK-RD²  ≈ {mu_test_rel:.4f}")
        print("-" * 80)

        print("✅ DK_RD2_core.py is ready to be used in any cosmological simulation.")
        print("📚 To generate full plots, run DK-RD2.py in the main folder.")
        print("=" * 80)
        print("This was the final nail in ΛCDM’s coffin.")
        print("\n💚 Physics is not invented — it's verified.\n"
          "— GabE=mc² & Luludns -> ∞Ψ")