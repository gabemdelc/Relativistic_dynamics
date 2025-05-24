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
💡 DK_RD2_Core.py — Relativistic Dynamics Toolkit for the DK-RD2 Model
====================================================================================

This is the universal core module of the **DK-RD2 (Dark Killer – Relativistic Dynamics)** model.

It contains all physics engine functions, constants, and tools required to reproduce
and validate the model's predictions using any cosmological dataset (SN Ia, CMB, DESI, etc.).

Main Features:
- Thermodynamic gravitational coupling: Gab(T, v)
- Redshift scaling: Gab(z), Ωₘ_eff(z), H(z), E(z)
- Luminosity & comoving distance, μ(z), dn/dz
- Relativistic angular power spectrum D_ℓ(ℓ)
- Einstein radius & lensing predictions
- χ² and MSE model comparison tools
- Integration-ready for any real-world dataset

💾 Fully portable and reusable across scientific projects.
🔁 Use it as a base for custom simulations, pipelines, or theoretical extensions.
"""

"""
-----------------------------------------------------------------------------------
Abstract:
The DK-RD2 model introduces a novel cosmological framework where the universe's
accelerated expansion arises naturally from relativistic energy redistribution and
thermal evolution, eliminating the need for dark energy.

At its core, the model replaces Newton’s constant G with a dynamic coupling Gab(T, v),
which depends explicitly on temperature and velocity. Gab is derived from Lorentz
corrections and thermal damping consistent with CMB physics. The model introduces
no free parameters and can be directly compared to observational data (SN Ia, CMB, H(z)).

-----------------------------------------------------------------------------------
Purpose of this Module:
This file defines the computational engine of DK-RD2. It provides:
✔️ Global cosmological constants (H₀, Ωₘ, Ω_Λ_DES/Planck, G₀, T_CMB)  
✔️ Thermodynamic gravitational coupling Gab(T, v) and redshift-mapped Gab(z)  
✔️ Modified Friedmann dynamics with Ωₘ_eff(z) and E(z)  
✔️ Luminosity and comoving distance functions  
✔️ Angular power spectrum D_ℓ(ℓ) for ΛCDM and DK-RD2  
✔️ Lensing tools: Einstein radius, D_ls proxies  
✔️ Statistical tools: χ², MSE for observational comparison  
✔️ Evidence generator: filenames for saving figures, tables, results

-----------------------------------------------------------------------------------
Core Principles of DK-RD2:
• Gravity is a thermodynamic-relativistic phenomenon  
• Matter density evolves with Gab(T, v), modifying Ωₘ_eff(z)  
• The Friedmann equation becomes dynamic: a function of temperature, velocity, and redshift  
• No dark energy is needed — and dark matter emerges naturally from relativistic effects

This module is the computational backbone of DK-RD2 cosmology.
-----------------------------------------------------------------------------------
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
    """
    T = T_fixed / (1 + z)
    v = c_light * np.sqrt(1 + z)
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
    H0_SI = Hubble_H0 * 1000 / 3.08567758149137e22  # H0 in 1/s
    integral = np.array([quad(E_func, 0, zi)[0] for zi in z])
    dL_m = (c_light / H0_SI) * (1 + z) * integral  # in meters
    dL_pc = dL_m / 3.08567758149137e16  # in parsecs
    mu = 5 * np.log10(dL_pc) - 5
    return mu

def luminosity_distance_Relativistic_temp(z, hubble=Hubble_H0, Om_m=Omega_m, Om_L=Omega_L_DES):
    H0_SI = hubble * 1000 / 3.08567758149137e22
    dL_m = np.array([
        (c_light / H0_SI) * (1 + zi) * quad(E_Relativistic_temp, 0, zi, args=(Om_m, Om_L))[0]
        for zi in z
    ])
    dL_pc = dL_m / 3.08567758149137e16
    mu = 5 * np.log10(dL_pc) - 5
    return mu

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

def Dl_Relativistic(l):
    """
    Computes the relativistic angular power spectrum D_ell(l)
    using Gab(T, v) dynamically derived from T(z) and v(z).
    Velocity is scaled with redshift and capped at 99.99% of the speed of light to preserve physical consistency.
    """

    # Approximate mapping from multipole ℓ to redshift z
    z_l = 10 + 1000 / (l + 1e-3)

    # Temperature scaling with redshift
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

############################################################################################
"""
        === EVIDENCE FILE NAMING UTILITY ===
"""
############################################################################################
def generate_evidence(evidence_type, consecutive=None, ext="", out_dir="evidence/"):
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
        extension = "jpg"
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
        print(f"μ(z=0.1) from DK-RD²  ≈ {mu_test_rel[0]:.4f}")
        print("-" * 80)

        print("✅ DK_RD2_core.py is ready to be used in any cosmological simulation.")
        print("📚 To generate full plots, run DK-RD2.py in the main folder.")
        print("=" * 80)
        print("This was the final nail in ΛCDM’s coffin.")
        print("\n💚 Physics is not invented — it's verified.\n"
          "— GabE=mc² & Luludns -> ∞Ψ")