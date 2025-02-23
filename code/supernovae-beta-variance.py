import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import quad

# === PARAMETER VALUES FOR COMPARISON ===
beta_original = 0.1
beta_CMB = 2e-6  # Best fit value from CMB analysis

# === FUNCTION FOR G_eff ===
def G_eff(T, v, G0=6.67430e-11, beta=0.1):
    return G0 * (1 + beta * (v**2 / 3e8**2))

# === LOAD SUPERNOVA DATA ===
sn_data = pd.read_csv("data/SCPUnion2_mu_vs_z.txt", sep=r'\s+', comment='#', header=None)
sn_data.columns = ["SN", "z", "mu", "mu_err"]

# === ΛCDM MODEL ===
def E_LCDM(z, Omega_m=0.3, Omega_L=0.7):
    return 1.0 / np.sqrt(Omega_m * (1 + z) ** 3 + Omega_L)

def luminosity_distance_LCDM(z, H0=70, Omega_m=0.3, Omega_L=0.7):
    c = 3e5  # km/s
    dL = np.array([quad(E_LCDM, 0, zi, args=(Omega_m, Omega_L))[0] for zi in z])
    dL = (c / H0) * (1 + z) * dL
    return 5 * np.log10(dL)

# === RELATIVISTIC DYNAMICS MODEL (ORIGINAL BETA) ===
def E_Relativistic(z, Omega_m=0.3, Omega_L=0.7, beta=beta_original):
    G_rel = G_eff(2.7 / (1 + z), 3e7 * np.sqrt(1 + z), beta=beta)
    return 1.0 / np.sqrt(Omega_m * (1 + z) ** 3 * (G_rel / 6.67430e-11) + Omega_L)

def luminosity_distance_Relativistic(z, H0=70, Omega_m=0.3, Omega_L=0.7, beta=beta_original):
    c = 3e5  # km/s
    dL = np.array([quad(E_Relativistic, 0, zi, args=(Omega_m, Omega_L, beta))[0] for zi in z])
    dL = (c / H0) * (1 + z) * dL
    return 5 * np.log10(dL)

# === RELATIVISTIC DYNAMICS MODEL (BEST BETA FROM CMB) ===
def luminosity_distance_Relativistic_CMB(z, H0=70, Omega_m=0.3, Omega_L=0.7):
    return luminosity_distance_Relativistic(z, H0, Omega_m, Omega_L, beta_CMB)

# === COMPUTE DISTANCES ===
mu_LCDM = luminosity_distance_LCDM(sn_data["z"])
mu_Relativistic_Original = luminosity_distance_Relativistic(sn_data["z"])
mu_Relativistic_CMB = luminosity_distance_Relativistic_CMB(sn_data["z"])

# === COMPUTE CHI-SQUARED AND MSE FOR SUPERNOVAS ===
chi2_LCDM = np.sum(((sn_data["mu"] - mu_LCDM) / sn_data["mu_err"])**2)
mse_LCDM = np.mean((sn_data["mu"] - mu_LCDM) ** 2)

chi2_Relativistic_Original = np.sum(((sn_data["mu"] - mu_Relativistic_Original) / sn_data["mu_err"])**2)
mse_Relativistic_Original = np.mean((sn_data["mu"] - mu_Relativistic_Original) ** 2)

chi2_Relativistic_CMB = np.sum(((sn_data["mu"] - mu_Relativistic_CMB) / sn_data["mu_err"])**2)
mse_Relativistic_CMB = np.mean((sn_data["mu"] - mu_Relativistic_CMB) ** 2)

# === PLOTTING ===
plt.figure(figsize=(10, 6))
plt.errorbar(sn_data["z"], sn_data["mu"], yerr=sn_data["mu_err"], fmt='o', label='Supernova Type Ia (Union2)')
plt.plot(sn_data["z"], mu_LCDM, label='ΛCDM Model', color='red', linestyle='dashed')
plt.plot(sn_data["z"], mu_Relativistic_Original, label=f'Relativistic Model (β={beta_original})', color='blue')
plt.plot(sn_data["z"], mu_Relativistic_CMB, label=f'Relativistic Model (β={beta_CMB})', color='green')
plt.xlabel('Redshift (z)')
plt.ylabel('Distance Modulus (μ)')
plt.title(f'Comparison with Observational Data\n'
          f'ΛCDM: χ²={chi2_LCDM:.2f}, MSE={mse_LCDM:.4f}\n'
          f'Relativistic Model (β={beta_original}): χ²={chi2_Relativistic_Original:.2f}, MSE={mse_Relativistic_Original:.4f}\n'
          f'Relativistic Model (β={beta_CMB}): χ²={chi2_Relativistic_CMB:.2f}, MSE={mse_Relativistic_CMB:.4f}')
plt.legend()
plt.grid()
plt.show()

# === PRINT CHI-SQUARED AND MSE VALUES ===
print(f"ΛCDM Model: χ²={chi2_LCDM:.2f}, MSE={mse_LCDM:.4f}")
print(f"Relativistic Model (β={beta_original}): χ²={chi2_Relativistic_Original:.2f}, MSE={mse_Relativistic_Original:.4f}")
print(f"Relativistic Model (β={beta_CMB}): χ²={chi2_Relativistic_CMB:.2f}, MSE={mse_Relativistic_CMB:.4f}")
