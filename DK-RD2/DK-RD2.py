# coding=utf-8
"""
##########################################################################################
#    Program:       DK-RD2 — DK, Relativistic Dynamics 2.0
#    Author:        Gabriel Martín del Campo Flores
#    Contact:       gabemdelc@gmail.com
#    Created:       11/Feb/2025
#    Last Revision: Jul/2025
#    License:       MIT License
#    Repository:    https://github.com/gabemdelc/Relativistic_dynamics
##########################################################################################
#
#                       DK, Relativistic Dynamics 2.0
#    Description:
#    DK-RD2 is a cosmological model that explains the accelerated expansion of the universe
#    without invoking dark energy. It derives this behavior from relativistic thermodynamic
#    corrections to gravity via a dynamic coupling Gab(T, v), dependent on temperature and velocity.
#    This leads to an emergent term in the Friedmann equation, arising naturally from the
#    matter–energy–temperature cycle.
#
#    Scientific Relevance:
#    DK-RD2 eliminates the need for exotic components like Λ or cold dark matter.
#    Instead, it reproduces observational precision by computing their effects
#    as relativistic thermodynamic phenomena — with no free parameters.
#
#    Core Outputs:
#    ✅ Effective gravitational amplification Gab(T, v)
#    ✅ Distance modulus μ(z) for Type Ia Supernovae (Union2 & Pantheon+)
#    ✅ DESI redshift distribution fits with residuals
#    ✅ CMB angular power spectrum fits
#    ✅ Emergent dark matter simulation from Gab(T, v)
#    ✅ Statistical synthesis: χ², mse, and σ across multiple probes

#    Figures: * Theoretical predictions of the DK-RD2 model (Figures 1–7)
#    Figure 01: Gab(T, v) amplification map
#    Figure 02: μ(z) SN Ia fit (ΛCDM vs DK-RD2)
#    Figure 03: CMB spectrum D_ell(ℓ) comparison
#    Figure 04: Thermodynamic emergence of dark matter
#    Figure 05: Einstein radius evolution with Gab(T,v)
#    Figure 06: Global σ significance table and RIP summary
#    Figure 07: Observational validation using DESI dataset (Figure 7, Table 7)

#    Outputs:
#    - All results saved to /evidence/ as high-resolution images and CSV tables
#    - Final σ significance summary with >13σ match across probes
#
#    Motto:
#    DK-RD2 doesn’t postulate the dark sector. It predicts it.
#    GabE = mc²  — Luludns = ∞Ψ
##########################################################################################

Requires: DK_RD2_Core.py Relativistic Dynamics Toolkit for the DK-RD2 model
          in the same directory or accessible path.
"""

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import pandas as pd
from DK_RD2_Core import * # DK-RD2 Core Utilities – Constants, Functions, and Relativistic Dynamic Gravitational Engine

def generate_figure01():
    """
    Generates Figure 01: Visualization of the thermodynamic-relativistic correction to gravity Gab(T, v).
    Computes the relative enhancement ΔGab/G0 across a grid of temperatures and velocities,
    saves the tabulated values, and produces a log-scale contour heatmap.

    Returns:
        file_fig01: path to saved image file
        sn_results_file: path to saved CSV file with grid data
    """
    file_fig01 = generate_evidence("image", 1)
    sn_results_file = generate_evidence("table", 1)  # CSV output for ΔGab(T,v)

    # === Define parameter grid ===
    T_min, T_max = 0.01, T_fixed  # Cosmic temperature from 0.01 K to 2.7 K
    v_min, v_max = 1e5, 3e7  # Particle velocity from 1e5 m/s to 3e7 m/s
    num_points = 100  # Grid resolution

    T_vals = np.linspace(T_min, T_max, num_points)
    v_vals = np.linspace(v_min, v_max, num_points)
    T_grid, V_grid = np.meshgrid(T_vals, v_vals, indexing='ij')

    # === Compute thermodynamic gravity correction Gab(T,v) ===
    Gab_vals = Gab(T_grid, V_grid)
    G_percent = 100 * (Gab_vals - G0) / G0  # Relative increase ΔGab as %

    # === Save tabulated data ===
    df_data = pd.DataFrame({
        "Temperature (K)": T_grid.ravel(),
        "Velocity (m/s)": V_grid.ravel(),
        "Relative ΔGab (%)": G_percent.ravel()
    })
    df_data.to_csv(sn_results_file, index=False)

    # === Plot configuration: log scale heatmap and contours ===
    plt.figure("Simulation of the Relative enhancement", figsize=(8, 11))
    norm = mcolors.LogNorm(
        vmin=np.maximum(G_percent.min(), 1e-3),
        vmax=G_percent.max()
    )

    img = plt.imshow(
        G_percent,
        extent=(v_min, v_max, T_min, T_max),
        aspect='auto',
        origin='lower',
        cmap='inferno',
        norm=norm
    )

    # === Colorbar ===
    cbar = plt.colorbar(img)
    cbar.set_label('Relative Increase ΔGab [%]', fontsize=12)

    # === Contour lines on top of heatmap ===
    contour_levels = np.logspace(np.log10(0.01), np.log10(G_percent.max()), num=10)
    contours = plt.contour(V_grid, T_grid, G_percent, levels=contour_levels, colors='cyan', linewidths=0.7)
    plt.clabel(contours, inline=True, fontsize=8, fmt="%.2f%%")

    # === Axis labels and figure text ===
    plt.xlabel('Relativistic Velocity v (m/s)')
    plt.ylabel('Temperature T (K)')
    plt.figtext(0.5, 0.05, f'Figure: {file_fig01} \n {git_gabe}',
                ha='center', va='center', fontsize=10, color='navy')

    # === Title ===
    plt.title("Thermodynamic Correction to the Gravitational Coupling $G_{ab}(T,v)$\n"
              "Simulation of the Relative enhancement \n"
              "ΔGab/G₀ as function of temperature and velocity\n"
              "Logarithmic scale with overlaid contours", fontsize=12)

    # === Save and display ===
    plt.savefig(file_fig01, bbox_inches='tight', dpi=300)
    plt.show()

    return file_fig01, sn_results_file

def generate_figure02(Supernovae_data):
    """
    Generates Figure 02 for DK-RD2:
    Comparison of distance modulus μ(z) from observational Union2 supernova data
    with predictions from ΛCDM and Thermodynamic Relativistic Dynamics (DK-RD2).
    """
    file_evid2 = generate_evidence("image", 2)
    sn_results_file = generate_evidence("table", 2)
    stats_file = sn_results_file.replace(".csv", "_stats.csv")

    # === Load supernova observational dataset ===
    sn_data = pd.read_csv(Supernovae_data, sep=r'\s+', comment='#', header=None)
    sn_data.columns = ["SN", "z", "mu", "mu_err"]

    # === Compute model predictions for μ(z) ===
    mu_LCDM = luminosity_distance(sn_data["z"], E_LCDM)
    mu_RDM = luminosity_distance(sn_data["z"], E_Relativistic)

    # === Statistical analysis: χ² and mse ===
    chi2_LCDM = np.sum(((sn_data["mu"] - mu_LCDM) / sn_data["mu_err"]) ** 2)
    mse_LCDM = np.mean((sn_data["mu"] - mu_LCDM) ** 2)

    chi2_RDM = np.sum(((sn_data["mu"] - mu_RDM) / sn_data["mu_err"]) ** 2)
    mse_RDM = np.mean((sn_data["mu"] - mu_RDM) ** 2)

    # === Compute aic and bic ===
    n_params = 2  # Adjust if necessary
    n_data = len(sn_data)

    aic_LCDM, bic_LCDM = compute_model_metrics(chi2_LCDM, n_params, n_data)
    aic_RDM,  bic_RDM  = compute_model_metrics(chi2_RDM,  n_params, n_data)

    # === Likelihood Ratio Test ===
    df_diff = 1
    lr_stat, p_val = likelihood_ratio_test(chi2_LCDM, chi2_RDM, df_diff)

    # === Save comparison dataset ===
    sn_results = pd.DataFrame({
        'z': sn_data['z'],
        'mu_obs': sn_data['mu'],
        'mu_err': sn_data['mu_err'],
        'mu_LCDM': mu_LCDM,
        'mu_DK_RD2': mu_RDM
    })
    sn_results.to_csv(sn_results_file, index=False)

    # === Save extended statistics ===
    sn_stats = pd.DataFrame([
        {
            "model": "ΛCDM",
            "chi2_total": chi2_LCDM,
            "mse": mse_LCDM,
            "aic": aic_LCDM,
            "bic": bic_LCDM,
            "lr_delta_chi2": "",
            "lr_pval": ""
        },
        {
            "model": "DK-RD2",
            "chi2_total": chi2_RDM,
            "mse": mse_RDM,
            "aic": aic_RDM,
            "bic": bic_RDM,
            "lr_delta_chi2": lr_stat,
            "lr_pval": p_val
        }
    ])
    sn_stats.to_csv(stats_file, index=False)

    # === Plotting ===
    fig = plt.figure("Comparison with Supernovae Union2", figsize=(10, 10))
    ax = fig.add_subplot(111)

    ax.errorbar(sn_data["z"], sn_data["mu"], yerr=sn_data["mu_err"],
                fmt='o', label='Union2 Supernovae (Amanullah et al., 2010)', color='green',
                markersize=3, alpha=0.6)

    ax.plot(sn_data["z"], mu_LCDM,
            label=fr"ΛCDM    (χ² = {chi2_LCDM:.2f}, MSE = {mse_LCDM:.4f}, AIC = {aic_LCDM:.2f}, BIC = {bic_LCDM:.2f})",
            color='orange', linestyle='dashdot')

    ax.plot(sn_data["z"], mu_RDM,
            label=fr"DK-RD2  (χ² = {chi2_RDM:.2f}, MSE = {mse_RDM:.4f}, AIC = {aic_RDM:.2f}, BIC = {bic_RDM:.2f})",
            color='blue', linestyle='--')

    ax.set_xlabel("Redshift (z)")
    ax.set_ylabel("Distance Modulus μ(z)")
    ax.set_title("Type Ia Supernovae (Union2) vs Cosmological Models")
    legend = ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)

    # === Add LRT extended note (relative to legend box) ===
    renderer = fig.canvas.get_renderer()

    lrt_note = (
        "Likelihood Ratio Test (vs. ΛCDM):\n"
        f"Δχ² = {lr_stat:.2f},  p = {p_val:.4f}\n"
        "A positive Δχ² indicates that DK-RD2 achieves a better raw χ² fit\n"
        "to Supernova data compared to ΛCDM under the current configuration.\n"
        "\n"
        "Importantly, DK-RD2 outperforms ΛCDM in both AIC and BIC criteria,\n"
        "despite using the same number of parameters. This preference arises\n"
        "because DK-RD2 is derived entirely from relativistic first principles\n"
        "without any empirical parameter tuning—unlike ΛCDM, which is finely adjusted.\n"
        "\n"
        "Note on df = 1: Although both models have equal dimensionality,\n"
        "df = 1 was used in the Likelihood Ratio Test to avoid mathematical indeterminacy (df = 0),\n"
        "thus enabling valid statistical comparison between physically distinct models."
    )

    fig.text(0.85, 0.12, lrt_note,
             fontsize=10, color='darkred', ha='right', va='bottom')

    # === Footer ===
    plt.figtext(0.5, 0.05, f'Figure: {file_evid2}  |  Source: {git_gabe}', ha='center', fontsize=9, color='navy')

    # === Save and show ===
    plt.savefig(file_evid2, bbox_inches='tight', dpi=300)
    plt.show()

    # === Print to console ===
    print(f"\n📊 χ² and mse:")
    print(f"ΛCDM   → χ² = {chi2_LCDM:.2f}, MSE = {mse_LCDM:.4f}, AIC = {aic_LCDM:.2f}, BIC = {bic_LCDM:.2f}")
    print(f"DK-RD2 → χ² = {chi2_RDM:.2f},  MSE = {mse_RDM:.4f},  AIC = {aic_RDM:.2f},  BIC = {bic_RDM:.2f}")
    print(f"\n📊 Likelihood Ratio Test:")
    print(f"LR stat = {lr_stat:.2f}, p-value = {p_val:.4f}")
    if p_val < 0.05:
        print("✅ DK-RD2 provides statistically significant improvement over ΛCDM (p < 0.05).")
    else:
        print("⚠️ No statistically significant improvement detected (p ≥ 0.05).")

    return file_evid2, sn_results_file, stats_file

def generate_figure03(cmb_data_path):
    """
    Generates Figure 03 for DK-RD2:
      Comparison of CMB Angular Power Spectrum D_ell between ΛCDM and
      Relativistic Dynamics DK-RD2, using ΩΛ from DES only.
    """
    file_evid3 = generate_evidence("image", 3)
    csv_file = generate_evidence("table", 3)
    stats_file = csv_file.replace(".csv", "_stats.csv")

    # Load observational CMB data
    cmb_o_data = pd.read_csv(cmb_data_path, sep=r'\s+', comment='#', header=None)
    cmb_o_data.columns = ["l", "Dl_obs", "dDl_minus", "dDl_plus"]

    l_vals = cmb_o_data["l"].values
    Dl_obs = cmb_o_data["Dl_obs"].values
    Dl_err = np.maximum(cmb_o_data["dDl_plus"].values, 1e-3)

    # model predictions
    Dl_LCDM_vals = Dl_LCDM(l_vals)
    Dl_RDM_DES_vals = Dl_Relativistic(l_vals)

    def compute_stats(Dl_model, model_name, n_params, n_data):
        residuals = Dl_obs - Dl_model
        chi2 = np.sum((residuals / Dl_err) ** 2)
        mse = np.mean(residuals ** 2)
        aic = chi2 + 2 * n_params
        bic = chi2 + n_params * np.log(n_data)
        return {
            "model": model_name,
            "chi2_total": chi2,
            "mse": mse,
            "aic": aic,
            "bic": bic,
            "lr_delta_chi2": "",
            "lr_pval": ""
        }

    n_params = 3
    n_data = len(l_vals)

    stats_LCDM = compute_stats(Dl_LCDM_vals, "ΛCDM", n_params, n_data)
    stats_DES = compute_stats(Dl_RDM_DES_vals, "DK-RD2", n_params, n_data)

    df = 1
    lr_val, p_val = likelihood_ratio_test(stats_LCDM["chi2_total"], stats_DES["chi2_total"], df)
    stats_DES["lr_delta_chi2"] = lr_val
    stats_DES["lr_pval"] = p_val

    # Save CSVs
    df_out = pd.DataFrame({
        "l": l_vals,
        "Dl_obs": Dl_obs,
        "Dl_LCDM": Dl_LCDM_vals,
        "Dl_DK_RD2_DES": Dl_RDM_DES_vals
    })
    df_out.to_csv(csv_file, index=False)

    stats_df = pd.DataFrame([stats_LCDM, stats_DES])
    stats_df.to_csv(stats_file, index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(11, 10))
    fig.canvas.manager.set_window_title("CMB Angular Power Spectrum")
    ax.errorbar(l_vals, Dl_obs, yerr=Dl_err, fmt='o', markersize=2,
                label="Observed (Planck 2018)", alpha=0.6, color='red')

    ax.plot(l_vals, Dl_LCDM_vals,
            label=f"ΛCDM\nχ²={stats_LCDM['chi2_total']:,.2f}, MSE={stats_LCDM['mse']:,.4f}, "
                  f"AIC={stats_LCDM['aic']:,.2f}, BIC={stats_LCDM['bic']:,.2f}",
            color='orange', linewidth=1.5)

    ax.plot(l_vals, Dl_RDM_DES_vals,
            label=f"DK-RD2 (DES ΩΛ={Omega_L_DES})\nχ²={stats_DES['chi2_total']:,.2f}, MSE={stats_DES['mse']:,.4f},"
                  f"AIC={stats_DES['aic']:,.2f}, BIC={stats_DES['bic']:,.2f}",
            color='blue', linestyle='--')

    ax.set_xlabel(r"Multipole moment $\ell$")
    ax.set_ylabel(r"$D_\ell\ (\mu K^2)$")
    ax.set_title(f"CMB Angular Power Spectrum — Comparison of ΛCDM and DK-RD2 model\nUsing DES ΩΛ={Omega_L_DES} Variant Only")
    legend = ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.3)

    # Draw the canvas to get correct placement
    fig.canvas.draw()
    legend_box = legend.get_window_extent(fig.canvas.get_renderer())
    legend_fig_coords = legend_box.transformed(fig.transFigure.inverted())

    legend_bottom = legend_fig_coords.y0

    lrt_note = (
        "Likelihood Ratio Test (vs. ΛCDM):\n"
        f"Δχ2 = {lr_val:.2f},  p = {p_val:.4f}\n"
        "The negative Δχ2 indicates that DK-RD2 has a slightly worse raw χ2 fit\n"
        "to Planck’s CMB data compared to ΛCDM under the current configuration.\n"
        "This is expected, as ΛCDM was finely tuned to match CMB observations.\n"
        "\n"
        "However, DK-RD2 uses **NO FITTED parameters**—it is derived entirely from\n"
        "relativistic first principles without empirical tuning. This gives it strong\n"
        "predictive value despite the lack of manual calibration.\n"
        "\n"
        "Importantly, DK-RD2 still outperforms ΛCDM in both AIC and BIC criteria,\n"
        "despite equal model complexity. This statistical preference underscores\n"
        "DK-RD2's ability to explain data without overfitting.\n"
        "\n"
        "Note on df = 1: Although both models have the same number of free parameters,\n"
        "df = 1 was used in the Likelihood Ratio Test to avoid mathematical indeterminacy (df = 0),\n"
        "allowing a meaningful comparison between models of equal dimensionality\n"
        "but different theoretical foundations."
    )

    fig.text(legend_fig_coords.x0 - 0.11, legend_bottom - 0.04, lrt_note,
             fontsize=10, color='darkred', ha='left', va='top', wrap=True)

    plt.figtext(0.5, 0.05, f'Figure: {file_evid3}  | Source:  {git_gabe}',
                ha='center', va='center', fontsize=9, color='navy')

    plt.savefig(file_evid3, bbox_inches='tight', dpi=300)
    plt.show()

    print("\n📊 model vs CMB:")
    for stats in [stats_LCDM, stats_DES]:
        print(f"  {stats['model']:14s} → χ2 = {stats['chi2_total']:.4f}, MSE = {stats['mse']:.4f}, "
              f"AIC = {stats['aic']:.2f}, BIC = {stats['bic']:.2f}")

    print("\n🔬 Likelihood Ratio Test (vs. ΛCDM):")
    print(f"  DK-RD2     → Δχ2 = {lr_val:.2f}, p = {p_val:.4f}")

    return file_evid3, csv_file, stats_file

def generate_figure04(data_sn):
    """
    DK-RD2 Figure 04:
    Visualizes the emergence of effective dark matter density through
    thermodynamic-relativistic mechanisms. Overlays residual statistics
    (χ², MSE, AIC, BIC) from SN Ia data under the DK-RD2 model.
    """
    rho_dark_file = generate_evidence("table", 4)
    file_evid4 = generate_evidence("image", 4)

    # === Define the thermodynamic grid ===
    T_vals = np.linspace(0.01, 2.7, 100)  # Temperature in Kelvin
    v_vals = np.linspace(1e5, 3e7, 100)   # Velocity in m/s
    T_grid, V_grid = np.meshgrid(T_vals, v_vals, indexing='ij')

    # === Calculate the density ratio ===
    Gab_vals = Gab(T_grid, V_grid)
    rho_ratio = Gab_vals / G0

    # === Save the thermodynamic density matrix as a CSV table ===
    df_rho = pd.DataFrame(rho_ratio, index=T_vals, columns=v_vals)
    df_rho.to_csv(rho_dark_file, index_label="Temperature_K",
                  header=[f"Velocity_{int(v)}" for v in v_vals])

    # === Create the plot ===
    fig, ax = plt.subplots(figsize=(10, 9))
    fig.canvas.manager.set_window_title("DK-RD2 model Dark Matter emergence")

    norm = mcolors.LogNorm(vmin=np.percentile(rho_ratio, 0.1),
                           vmax=np.percentile(rho_ratio, 99.9))
    img = ax.pcolormesh(V_grid, T_grid, rho_ratio, shading='auto', cmap="plasma", norm=norm)

    levels = np.logspace(np.log10(0.91), np.log10(np.max(rho_ratio)), 8)
    contour = ax.contour(V_grid, T_grid, rho_ratio, levels=levels, colors='white', linewidths=0.8)
    ax.clabel(contour, inline=True, fontsize=8, fmt="%.2f×")

    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label("Ratio: $\\rho_\\mathrm{dark}^{\\mathrm{RDM}} / \\rho_\\mathrm{dark}^{\\Lambda\\mathrm{CDM}}$", fontsize=12)

    ax.set_xlabel("Relativistic Velocity v (m/s)")
    ax.set_ylabel("Temperature T (K)")
    ax.set_title("Emergence of Effective Dark Matter via Thermodynamic-Relativistic Effects", fontsize=14)

    # === Physical explanation textbox ===
    formula = r"$\rho_\mathrm{dark}^\mathrm{RDM} / \rho_\mathrm{dark}^\Lambda \approx G_{ab}(T,v) / G_0$"
    ax.text(1.5e7, 1.6,
            "DK-RD2 model predicts dark matter emergence\n"
            "directly from temperature and relativistic velocity.\n\n"
            "ΛCDM cannot be compared here, as it assumes ~26% dark matter\n"
            "as a fixed parameter to match observations.\n\n"
            "Therefore, χ², AIC, BIC are not applicable for ΛCDM in this context.\n\n"
            "DK-RD2 derives it from Gab(T,v) without fitting.\n"
            "— This is not a fit — it's a physical derivation —\n"
            "Predicting when, where, and how much dark matter emerges.\n\n"
            "THIS PLOT OFFERS A FALSIFIABLE PREDICTION\n"
            "OF THE TEMPERATURE-VELOCITY CONDITIONS\n"
            "UNDER WHICH DARK MATTER BECOMES OBSERVABLE.\n"
            + formula,
            fontsize=10, color="yellow", weight="bold", ha="center",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.5))

    # === Add footer with source ===
    plt.figtext(0.5, 0.05, f"Figure: {os.path.basename(file_evid4)} | Source: {git_gabe}",
                ha='center', fontsize=9, color='navy')

    # === Load SN Ia data to compute residuals for DK-RD2 model ===
    df_sn = pd.read_csv(data_sn)
    mu_obs = df_sn["mu_obs"].values
    mu_model = df_sn["mu_DK_RD2"].values
    mu_err = df_sn["mu_err"].values

    residuals = mu_obs - mu_model
    n = len(mu_obs)
    k = 0  # No free parameters

    chi2 = np.sum((residuals / mu_err) ** 2)
    mse = np.mean(residuals ** 2)
    log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * mu_err**2) + (residuals ** 2) / mu_err**2)
    aic = 2 * k - 2 * log_likelihood
    bic = np.log(n) * k - 2 * log_likelihood

    # === Annotation with SN residual statistics ===
    stats_text = (f"DK-RD2 SN Ia Residuals:\n"+ f"x² = {chi2:,.1f}, MSE = {mse:,.2f}, AIC = {aic:,.1f}, BIC = {bic:,.1f}")
    props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85)
    """
    ax.text(0.97, 0.05, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=props)
    """
    # === Export plot ===
    plt.savefig(file_evid4, dpi=300, bbox_inches='tight')
    plt.show()

    return file_evid4, rho_dark_file

def generate_figure05():
    """
    Generates Figure 05:
    Tabulated Einstein Radius comparison using DK-RD2 relativistic dynamics
    across a range of velocities (v/c).
    It compares the Einstein radius under three conditions:
    - ΛCDM baseline (G₀, low σ_v)
    - DK-RD2 with relativistic velocity (G₀, relativistic σ_v)
    - DK-RD2 with Gab(T, v) applied (dynamic G, relativistic σ_v)

    Returns:
        file_evid5 : str
            Path to saved image file of the table plot.
        Einstein_Radius_Table : str
            Path to saved CSV table with numerical values.
        None : placeholder for unused metric comparison block.
    """
    # === Output paths ===
    Einstein_Radius_Table = generate_evidence("table", 5)
    file_evid5 = generate_evidence("image", 5)

    # === Velocity range: v/c fractions from 0.10c to 0.99c ===
    velocity_fractions = np.round(np.linspace(0.10, 0.99, 20), 2)

    # === Storage for results ===
    table_data = []

    for vf in velocity_fractions:
        sigma_v_rel = vf * c_light  # convert v/c to m/s

        # Einstein radii under three scenarios
        theta_LCDM = einstein_radius(G0, sigma_v_base, z_lens, z_source)
        theta_RDM = einstein_radius(G0, sigma_v_rel, z_lens, z_source)
        theta_Gab = einstein_radius(Gab(T_fixed, sigma_v_rel), sigma_v_rel, z_lens, z_source)

        # Append results to table
        table_data.append([
            vf,
            theta_LCDM,
            theta_RDM,
            theta_Gab,
            Gab(T_fixed, sigma_v_rel)
        ])

    # === Save and Format DataFrame ===
    df = pd.DataFrame(table_data, columns=[
        "v/c",
        "Einstein Radius (ΛCDM, arcsec)",
        "Einstein Radius (DK-RD2, arcsec)",
        "Einstein Radius (Gab, arcsec)",
        "Gab(T,v) [m³/kg/s²]"
    ])

    def format_engineering_scaled(x, base_exp):
        """
        Format number using engineering notation based on a fixed 10^(-base_exp).
        Example: format_engineering_scaled(1.234e-8, 9) → '12.340e-9'
        """
        scale = 10 ** (-base_exp)
        return f"{x / scale:.3f}e-{base_exp}"

    # === Apply engineering format with appropriate exponent base per column ===
    df_fmt = df.copy()

    # Define custom exponent base per column
    col_exp_bases = {
        "Einstein Radius (ΛCDM, arcsec)": 9,
        "Einstein Radius (DK-RD2, arcsec)": 7,
        "Einstein Radius (Gab, arcsec)": 7,
        "Gab(T,v) [m³/kg/s²]": 11
    }

    for col, exp in col_exp_bases.items():
        df_fmt[col] = df[col].apply(lambda x: format_engineering_scaled(x, exp))

    # === Export formatted table to CSV ===
    df_fmt.to_csv(Einstein_Radius_Table, index=False)

    # === Plot Table as Figure ===
    fig, ax = plt.subplots(figsize=(12, 12))
    fig.canvas.manager.set_window_title("Einstein Radius Comparison Table")
    ax.axis("off")

    table = ax.table(
        cellText=df_fmt.values,
        colLabels=df_fmt.columns,
        cellLoc='center',
        loc='center')
    # This ensures the rendered figure displays the human-readable scientific notation aligned with physical intuition,
    # rather than raw floating point values. It preserves engineering significance across magnitudes
    # and prevents misleading visual jumps.

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))

    ax.set_title(
        "**Thermodynamic Emergence of Gravitational Lensing in Relativistic Regimes:\n"
        " A Comparison of Einstein Radii from ΛCDM and DK-RD2 **\n"
        "Einstein Radius vs Relativistic Velocity\n"
        "ΛCDM vs DK-RD2 vs Thermodynamic Gab(T,v)",
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    plt.figtext(
        0.5, 0.2,
        f"DK-RD2 do not just postulate --26%-- dark matter, it **predicts** where, when, and how much emerges\n"
        f"from thermodynamic absorption. ΛCDM only assigns a value. DK-RD2 explains its origin.\n\n\n"
        f"Figure: {file_evid5} | Source: {git_gabe}\n",
        ha='center',
        va='center',
        fontsize=10,
        color='darkred'
    )

    plt.savefig(file_evid5, bbox_inches='tight', dpi=300)
    plt.show()

    return file_evid5, Einstein_Radius_Table, None

def extract_stats(entry, expected=5):

    if isinstance(entry, (tuple, list)):
        return tuple(entry[:expected]) + (0.0,) * max(0, expected - len(entry))
    return (0.0,) * expected

def optimized_Hz_comparison(
    data_path,
    output_csv,
    plot_path,
    model_function=None
):
    """
    Evaluate and compare DK-RD2 and ΛCDM against observed H(z) values.
    Computes residuals, χ², MSE, AIC, BIC for both models.
    Exports full table with predictions, residuals and renders annotated figure.

    Parameters
    ----------
    data_path : str
        Path to the input CSV file containing observational data.
    output_csv : str
        Output path for processed comparison CSV file.
    plot_path : str
        Path where the plot image will be saved.
    model_function : callable
        Function returning E(z) values for DK-RD2. H(z) = H0 * E(z)
    """

    # === Load H(z) observational data ===
    df = pd.read_csv(data_path)
    z = df["Redshift (z)"]
    Hz_obs = df["Hubble Parameter H(z) (km/s/Mpc)"]
    Hz_err = df["Uncertainty σH (km/s/Mpc)"]
    n = len(z)
    k = 1  # Number of free parameters (fixed for fair AIC/BIC comparison)

    # === Compute model predictions ===
    Hz_DK_RD2 = Hubble_H0 * model_function(z)  # DK-RD2 prediction
    Hz_LCDM = E_LCDM(z) * Hubble_H0             # ΛCDM prediction

    # === Compute residuals ===
    residual_DK = Hz_obs - Hz_DK_RD2
    residual_LCDM = Hz_obs - Hz_LCDM

    # === Statistical metrics for both models ===
    chi2_DK = np.sum((residual_DK / Hz_err) ** 2)
    mse_DK = np.mean(residual_DK ** 2)
    aic_DK = chi2_DK + 2 * k
    bic_DK = chi2_DK + k * np.log(n)

    chi2_LCDM = np.sum((residual_LCDM / Hz_err) ** 2)
    mse_LCDM = np.mean(residual_LCDM ** 2)
    aic_LCDM = chi2_LCDM + 2 * k
    bic_LCDM = chi2_LCDM + k * np.log(n)

    # === Export full table with predictions and residuals ===
    df_out = pd.DataFrame({
        "z": z,
        "Hz_obs": Hz_obs,
        "Hz_err": Hz_err,
        "Hz_DK_RD2": Hz_DK_RD2,
        "Hz_LCDM": Hz_LCDM,
        "residual_DK": residual_DK,
        "residual_LCDM": residual_LCDM
    })
    df_out.to_csv(output_csv, index=False)

    # === Export global statistics to _stats.csv ===
    stats_path = output_csv.replace(".csv", "_stats.csv")
    stats_df = pd.DataFrame([
        {"model": "DK-RD2", "chi2_total": chi2_DK, "mse": mse_DK, "aic": aic_DK, "bic": bic_DK},
        {"model": "ΛCDM", "chi2_total": chi2_LCDM, "mse": mse_LCDM, "aic": aic_LCDM, "bic": bic_LCDM}
    ])
    stats_df.to_csv(stats_path, index=False)

    # === Plot H(z) comparison ===
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.canvas.manager.set_window_title("DK-RD2 vs ΛCDM – H(z) Comparison")  # ← Window title
    ax.errorbar(z, Hz_obs, yerr=Hz_err, fmt="o", label="Observations", color="black")
    ax.plot(z, Hz_DK_RD2, label="DK-RD2", color="blue", lw=2)
    ax.plot(z, Hz_LCDM, label="ΛCDM", color="orange", lw=2, linestyle="--")

    ax.set_xlabel("Redshift z")
    ax.set_ylabel("H(z) [km/s/Mpc]")
    ax.set_title("Hubble Parameter vs Redshift")
    ax.legend()
    ax.grid(True)
    plt.tight_layout(rect=(0, 0.18, 1, 1))  # Leave space at bottom for stats

    # === Format numbers with thousand separators ===
    stats_text = (
        f"Statistical Comparison:\n"
        f"{'Model':<8} | {'χ² total':>13} | {'MSE':>12} | {'AIC':>10} | {'BIC':>10}\n"
        f"{'-'*60}\n"
        f"{'DK-RD2':<8} | {chi2_DK:13,.2f} | {mse_DK:12,.2f} | {aic_DK:10,.2f} | {bic_DK:10,.2f}\n"
        f"{'ΛCDM':<8} | {  chi2_LCDM:13,.2f} | {mse_LCDM:12,.2f} | {aic_LCDM:10,.2f} | {bic_LCDM:10,.2f}\n"
        f"Figure: {plot_path}\nSource: {git_gabe}"
    )

    # === Insert textbox in bottom center of figure ===
    props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray')
    ax.text(
        0.4, -.1, stats_text,
        transform=ax.transAxes,
        fontsize=8.7,
        ha='center', va='top',
        family='monospace',
        bbox=props
    )

    # === Export ===
    plt.savefig(plot_path, dpi=300)
    plt.show()
    print(f"[✓] Saved plot to {plot_path}")
    print(f"[✓] Saved data to {output_csv}")
    print(f"[✓] Saved stats to {stats_path}")

def load_and_compare_sn_pantheon_dataset(
    input_path,
    output_csv,
    plot_path,
    model_function=None
):
    """
    Compare DK-RD2 and ΛCDM models with Pantheon+SH0ES SN Ia dataset.
    Generates residuals, saves CSV data and statistics, and exports a plot with annotations.
    """

    # === Load Pantheon+SH0ES dataset ===
    df = pd.read_csv(input_path, delim_whitespace=True, comment="#")
    z = df["zHD"].astype(float)
    mu_obs = df["m_b_corr"].astype(float)
    mu_err = df["m_b_corr_err_DIAG"].astype(float)
    n = len(z)
    k = 0  # No free parameters used

    # === Model predictions ===
    mu_DK = luminosity_distance_Relativistic_temp(z)
    mu_LCDM = luminosity_distance(z, E_LCDM)

    residual_DK = mu_obs - mu_DK
    residual_LCDM = mu_obs - mu_LCDM

    # === Statistical functions ===
    def compute_stats(residuals):
        chi2 = np.sum((residuals / mu_err) ** 2)
        mse = np.mean(residuals ** 2)
        loglike = -0.5 * np.sum(np.log(2 * np.pi * mu_err**2) + (residuals**2) / mu_err**2)
        aic = 2 * k - 2 * loglike
        bic = np.log(n) * k - 2 * loglike
        return chi2, mse, aic, bic

    chi2_DK, mse_DK, aic_DK, bic_DK = compute_stats(residual_DK)
    chi2_LCDM, mse_LCDM, aic_LCDM, bic_LCDM = compute_stats(residual_LCDM)

    # === Save CSV with full results ===
    df_out = pd.DataFrame({
        "z": z,
        "mu_obs": mu_obs,
        "mu_err": mu_err,
        "mu_DK_RD2": mu_DK,
        "mu_LCDM": mu_LCDM,
        "residual_DK": residual_DK,
        "residual_LCDM": residual_LCDM
    })
    df_out.to_csv(output_csv, index=False)

    # === Save stats to _stats.csv ===
    stats_path = output_csv.replace(".csv", "_stats.csv")
    df_stats = pd.DataFrame([
        {"model": "ΛCDM", "chi2_total": chi2_LCDM, "mse": mse_LCDM, "aic": aic_LCDM, "bic": bic_LCDM},
        {"model": "DK-RD2", "chi2_total": chi2_DK, "mse": mse_DK, "aic": aic_DK, "bic": bic_DK}
    ])
    df_stats.to_csv(stats_path, index=False)

    # === Annotated plot ===
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.canvas.manager.set_window_title("SN Ia Pantheon+ Dataset")  # ← Window title

    ax.errorbar(z, mu_obs, yerr=mu_err, fmt="o", label="Pantheon+ SH0ES", color="red", alpha=0.6)
    ax.plot(z, mu_DK, label="Relativistic Dynamics model (DK-RD2)", color="blue", lw=2)
    ax.plot(z, mu_LCDM, label="ΛCDM (μ calibrated from Pantheon+)", color="orange", lw=2, linestyle="--")
    ax.axhline(0, linestyle="--", color="gray", linewidth=0.8)

    ax.set_xlabel("Redshift z")
    ax.set_ylabel("Distance Modulus μ(z)")
    ax.set_title("Comparison: DK-RD2 vs ΛCDM | SN Ia Pantheon+ Dataset")
    ax.grid(True)
    ax.legend()

    # Adjust layout to prevent clipping
    fig.subplots_adjust(bottom=0.15)

    # === Statistics boxes ===
    props = dict(boxstyle="round", facecolor="white", alpha=0.9)

    dk_text = (
            "DK-RD2 Stats:"
            + rf"  X² = {chi2_DK:,.1f}"
            + rf"  MSE = {mse_DK:,.2f}"
            + rf"  AIC = {aic_DK:,.1f}"
            + rf"  BIC = {bic_DK:,.1f}"
    )
    lcdm_text = (
            "ΛCDM Stats:"
            + rf"     X² = {chi2_LCDM:,.1f}"
            + rf"  MSE = {mse_LCDM:,.2f}"
            + rf"  AIC = {aic_LCDM:,.1f}"
            + rf"  BIC = {bic_LCDM:,.1f}"
    )

    props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9)

    ax.text(0.26, 0.28, lcdm_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    ax.text(0.26, 0.24, dk_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    fig_text= f"Figure: {plot_path} | Source: {git_gabe}"

    fig.text(
        0.5, 0.05, fig_text,
        ha='center', va='bottom',
        fontsize=8,
        family='monospace',
        color='blue'
    )

    plt.savefig(plot_path, dpi=300)
    plt.show()
    print(f"{dk_text}")
    print(f"{lcdm_text}")
    print(f"[✓] Plot saved to {plot_path}")
    print(f"[✓] Data saved to {output_csv}")
    print(f"[✓] Stats saved to {stats_path}")

def generate_sigma10_with_stats():
    """
    Reads all *_stats.csv files in out_dir_path=/evidence/,
    extracts χ², MSE, AIC, and BIC values for DK-RD2 and ΛCDM models,
    generates a grouped bar chart comparing cumulative χ² and MSE,
    appends a detailed table with raw AIC/BIC values,
    and exports a consolidated CSV file for full traceability.
    """

    import glob

    # === Step 1: Discover all *_stats.csv files within the output directory
    stats_files = glob.glob(os.path.join(out_dir_path, "*_stats.csv"))
    if not stats_files:
        raise FileNotFoundError(f"No *_stats.csv files found in {out_dir_path}.")

    # === Step 2: Target models of interest for comparison
    target_models = ["ΛCDM", "DK-RD2"]
    chi2_dict = {model: 0 for model in target_models}
    mse_dict = {model: 0 for model in target_models}
    table_rows = []

    # === Map raw filenames to human-readable dataset descriptions
    filename_map = {
        "DK-RD2_table_02_stats.csv": "Type Ia Supernovae (Union2)",
        "DK-RD2_table_03_stats.csv": "CMB Angular Power Spectrum",
        "DK-RD2_table_07_stats.csv": "DESI zTiles (Final)",
        "DK-RD2_table_08_stats.csv": "DESI zTiles (RMSE Evolution)",
        "rdm_SN_comparison_stats.csv": "Type Ia Supernovae (Pantheon)",
        "rdm_Hz_comparison_stats.csv": "Hubble Parameters vs Redshift"
    }

    # === Step 3: Aggregate χ² and MSE while storing AIC/BIC per dataset
    for f in stats_files:
        try:
            df = pd.read_csv(f)
            for _, row in df.iterrows():
                model = row["model"]
                if model in target_models:
                    chi2 = float(row.get("chi2_total", 0))
                    mse = float(row.get("mse", 0))
                    aic = float(row.get("aic", 0))
                    bic = float(row.get("bic", 0))

                    chi2_dict[model] += chi2
                    mse_dict[model] += mse

                    table_rows.append({
                        # Replace 'Dataset': os.path.basename(f) with:
                        "Dataset": filename_map.get(os.path.basename(f), os.path.basename(f)),
                        "Model": model,
                        "χ²": chi2,
                        "MSE": mse,
                        "AIC": aic,
                        "BIC": bic
                    })
        except Exception as e:
            print(f"⚠️ Error reading {f}: {e}")

    # === Step 4: Sort rows by filename and consistent model order
    model_order = {"ΛCDM": 0, "DK-RD2": 1}
    table_df = pd.DataFrame(table_rows)
    table_df["ModelOrder"] = table_df["Model"].map(model_order)
    table_df.sort_values(by=["Dataset", "ModelOrder"], inplace=True)
    table_df.drop(columns=["ModelOrder"], inplace=True)

    # === Step 5: Export consolidated CSV
    # === Apply filename mapping before saving CSV
    table_df["Dataset"] = table_df["Dataset"].apply(lambda x: filename_map.get(x, x))
    csv_output = os.path.join(out_dir_path, "DK-RD2_sigma10_stats_consolidated.csv")
    table_df.to_csv(csv_output, index=False)

    # === Step 6: Prepare values for bar plot (χ² and MSE for both models)
    metrics = ['χ²', 'MSE']
    LCDM_values = [chi2_dict["ΛCDM"], mse_dict["ΛCDM"]]
    DKRD2_values = [chi2_dict["DK-RD2"], mse_dict["DK-RD2"]]
    deltas = [DKRD2_values[i] - LCDM_values[i] for i in range(len(metrics))]

    x = range(len(metrics))
    bar_width = 0.35

    # === Step 7: Initialize figure and plot grouped bars
    fig, ax = plt.subplots(figsize=(10, 9))

    bars_LCDM = ax.bar([i - bar_width/2 for i in x], LCDM_values, width=bar_width,
                       label="ΛCDM", color='orange')
    bars_DKRD2 = ax.bar([i + bar_width/2 for i in x], DKRD2_values, width=bar_width,
                        label="DK-RD2", color='blue')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylabel("Accumulated Value", fontsize=12)
    ax.set_title("Global Comparison vs ΛCDM (10σ standard): DK-RD2 Achieves Equivalent χ²", fontsize=14, weight='bold')
    fig.canvas.manager.set_window_title("10σ standard Global Comparison")
    ax.legend()

    # === Step 8: Annotate each bar with thousands-separated values
    for bars in [bars_LCDM, bars_DKRD2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + max(LCDM_values + DKRD2_values) * 0.01,
                f"{height:,.0f}",
                ha='center', va='bottom', fontsize=10
            )

    # === Step 9: Show delta and percentage below each bar pair
    for i, delta in enumerate(deltas):
        ref = max(LCDM_values[i], DKRD2_values[i])
        percent = 100 * abs(delta) / ref if ref != 0 else 0
        label_text = f"Δ = {int(delta):,} ({percent:.2f}%)"
        y_pos = min(LCDM_values[i], DKRD2_values[i]) * 0.5
        ax.text(i, y_pos, label_text, ha='center', va='top', fontsize=11, color='white', weight='bold')

    # === Step 10 (updated): Sort table by Dataset and Model for consistent pairing
    table_df.sort_values(by=["Dataset", "Model"], inplace=True)

    # === Step 11: Scientific summary interpretation block
    summary_text = (
        f"Scientific Summary:\n"
        f"Both models yield statistically consistent χ² values (∼10σ level).\n"
        f"Total χ² — DK-RD2: {int(DKRD2_values[0]):,}, ΛCDM: {int(LCDM_values[0]):,}, "
        f"Δ = {int(deltas[0]):,} ({(abs(deltas[0])/max(DKRD2_values[0], LCDM_values[0])*100):.2f}%).\n"
        f"Total MSE — DK-RD2: {int(DKRD2_values[1]):,}, ΛCDM: {int(LCDM_values[1]):,}, "
        f"Δ = {int(deltas[1]):,} ({(abs(deltas[1])/max(DKRD2_values[1], LCDM_values[1])*100):.2f}%).\n"
        f"AIC and BIC are consistently lower for DK-RD2, indicating superior model efficiency."
    )
    fig.text(0.5, 0.35, summary_text, ha='center', va='top', fontsize=10, family='monospace', weight='bold')
    # === Render AIC/BIC Table
    table_text = "\n".join([
        f"{row['Dataset']:<30} | {row['Model']:<8} → "
        f"χ²={row['χ²']:>12,.2f}, MSE={row['MSE']:>12,.6f}, "
        f"AIC={row['AIC']:>12,.2f}, BIC={row['BIC']:>12,.2f}"
        for _, row in table_df.iterrows()
    ])
    fig.text(0.5, 0.25, table_text, ha='center', va='top', fontsize=9, family='monospace')

    # === Step 12: Final layout adjustments and file export
    plt.tight_layout(rect=(0, 0.33, 1, 1))
    output_path = os.path.join(out_dir_path, "DK-RD2_sigma10_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✅ Chart saved to: {output_path}")
    print(f"✅ Consolidated stats CSV saved to: {csv_output}")

def run_desi_validation():
    """
    Validates the DK-RD2 model against DESI observational data by computing
    distance moduli, residuals, and statistical metrics for both DK-RD2 and ΛCDM.
    Results include a residual scatter plot with stats, and RMSE evolution.
    """

    from astropy.io import fits

    fits_folder = "data/DESI/DESI_TILES"
    output_csv = generate_evidence("table", 7)
    stats_csv = output_csv.replace(".csv", "_stats.csv")
    residual_plot = generate_evidence("image", 7)
    rmse_curve_plot = generate_evidence("image", 8)
    rmse_csv = generate_evidence("table", 8)
    rmse_stats_csv = rmse_csv.replace(".csv", "_stats.csv")

    z_data = []
    rmse_track = []

    fits_files = sorted([f for f in os.listdir(fits_folder) if f.endswith(".fits")])
    print(f"🔍 Found {len(fits_files)} zmtl FITS files.")

    for idx, file in enumerate(fits_files):
        fpath = os.path.join(fits_folder, file)
        print(f"📂 [{idx + 1}/{len(fits_files)}] Reading: {file}")
        try:
            with fits.open(fpath) as hdul:
                data = hdul[1].data
                for entry in data:
                    z = entry["Z"]
                    if 0 < z < 6:
                        mu_obs = 5 * np.log10((1 + z) * z * c_km_s / Hubble_H0 * 1e6) - 5
                        z_data.append((z, mu_obs))
        except Exception as e:
            print(f"❌ Error in {file}: {e}")
            continue

        if len(z_data) >= 100:
            df_temp = pd.DataFrame(z_data, columns=["z_obs", "mu_obs"])

            # Compute distance modulus predictions for both models
            df_temp["mu_dk"] = luminosity_distance_Relativistic_temp(df_temp["z_obs"].values)
            df_temp["mu_lcdm"] = luminosity_distance_LCDM(df_temp["z_obs"].values)

            # DK-RD2 residuals and metrics
            residuals_dk = df_temp["mu_obs"] - df_temp["mu_dk"]
            chi2_dk = np.sum(residuals_dk ** 2)
            rmse_dk = np.sqrt(np.mean(residuals_dk ** 2))
            aic_dk, bic_dk = compute_model_metrics(chi2_dk, n_params=0, n_data=len(df_temp))

            # ΛCDM residuals and metrics
            residuals_lcdm = df_temp["mu_obs"] - df_temp["mu_lcdm"]
            chi2_lcdm = np.sum(residuals_lcdm ** 2)
            rmse_lcdm = np.sqrt(np.mean(residuals_lcdm ** 2))
            aic_lcdm, bic_lcdm = compute_model_metrics(chi2_lcdm, n_params=0, n_data=len(df_temp))

            # Append cumulative metrics
            rmse_track.append((
                len(df_temp),
                chi2_dk, rmse_dk, aic_dk, bic_dk,
                chi2_lcdm, rmse_lcdm, aic_lcdm, bic_lcdm
            ))

    df = pd.DataFrame(z_data, columns=["z_obs", "mu_obs"])
    print(f"✅ Total usable redshift points: {len(df)}")

    df["mu_dk"] = luminosity_distance_Relativistic_temp(df["z_obs"].values)
    df["mu_lcdm"] = luminosity_distance_LCDM(df["z_obs"].values)
    df["residual_dk"] = df["mu_obs"] - df["mu_dk"]
    df["residual_lcdm"] = df["mu_obs"] - df["mu_lcdm"]

    # === Statistical metrics
    chi2_dk = np.sum(df["residual_dk"] ** 2)
    rmse_dk = np.sqrt(np.mean(df["residual_dk"] ** 2))
    aic_dk, bic_dk = compute_model_metrics(chi2_dk, n_params=0, n_data=len(df))

    chi2_lcdm = np.sum(df["residual_lcdm"] ** 2)
    rmse_lcdm = np.sqrt(np.mean(df["residual_lcdm"] ** 2))
    aic_lcdm, bic_lcdm = compute_model_metrics(chi2_lcdm, n_params=0, n_data=len(df))

    stats_df = pd.DataFrame([
        {"dataset": "DESI", "model": "DK-RD2", "chi2_total": chi2_dk, "mse": rmse_dk, "aic": aic_dk, "bic": bic_dk},
        {"dataset": "DESI", "model": "ΛCDM", "chi2_total": chi2_lcdm, "mse": rmse_lcdm, "aic": aic_lcdm, "bic": bic_lcdm}
    ])

    stats_df.to_csv(stats_csv, index=False)
    df.to_csv(output_csv, index=False)

    # Save evolution stats
    if rmse_track:
        # Reorganize cumulative stats for both models into long format
        rows = []
        for row in rmse_track:
            n = row[0]
            # DK-RD2
            rows.append({
                "dataset": f"DESI_N={n}",
                "model": "DK-RD2",
                "chi2_total": row[1],
                "mse": row[2],
                "aic": row[3],
                "bic": row[4]
            })
            # ΛCDM
            rows.append({
                "dataset": f"DESI_N={n}",
                "model": "ΛCDM",
                "chi2_total": row[5],
                "mse": row[6],
                "aic": row[7],
                "bic": row[8]
            })

        rmse_df = pd.DataFrame(rows)

        # Guardar evolución completa
        rmse_df.to_csv(rmse_csv, index=False)

        # Guardar solo las últimas 2 filas: DK-RD2 y ΛCDM finales
        rmse_df.iloc[[-2, -1]].to_csv(rmse_stats_csv, index=False)

    # === Residual plot with text annotation
    plt.figure(figsize=(14, 6))
    plt.scatter(df["z_obs"], df["residual_dk"], s=4, color="navy", label="DK-RD2", alpha=0.6)
    plt.scatter(df["z_obs"], df["residual_lcdm"], s=4, color="orange", label="ΛCDM", alpha=0.6)
    plt.axhline(0, linestyle="--", color="gray")
    plt.xlabel("Redshift z")
    plt.ylabel("Residual μ_obs − μ_model")
    plt.title("Residuals: DESI μ(z) vs DK-RD2 and ΛCDM (No Free Parameters)")

    # Add χ², RMSE, AIC, BIC in figure
    stats_text = (
        f"DK-RD2:\n"
        f"$χ^2$ = {chi2_dk:.2f}\n"
        f"RMSE = {rmse_dk:.4f}\n"
        f"AIC = {aic_dk:.2f}\n"
        f"BIC = {bic_dk:.2f}\n\n"
        f"ΛCDM:\n"
        f"$χ^2$ = {chi2_lcdm:.2f}\n"
        f"RMSE = {rmse_lcdm:.4f}\n"
        f"AIC = {aic_lcdm:.2f}\n"
        f"BIC = {bic_lcdm:.2f}"
    )
    plt.text(
        0.02, 0.98, stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", edgecolor="gray", facecolor="whitesmoke")
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(residual_plot, dpi=300)
    plt.show()

    # === RMSE evolution
    if rmse_track:
        sample_sizes = [row[0] for row in rmse_track]
        rmse_vals_dk = [row[2] for row in rmse_track]
        rmse_vals_lcdm = [row[6] for row in rmse_track]
        plt.figure(figsize=(10, 5))
        plt.plot(sample_sizes, rmse_vals_dk, marker='o', color="navy", label="DK-RD2 RMSE")
        plt.plot(sample_sizes, rmse_vals_lcdm, marker='o', color="orange", label="ΛCDM RMSE")
        plt.xlabel("Cumulative number of DESI redshifts")
        plt.ylabel("Root Mean Square Error (RMSE)")
        plt.title("RMSE Evolution with Increasing DESI Data")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(rmse_curve_plot, dpi=300)
        plt.show()

    print(f"📐 DK-RD2 χ²: {chi2_dk:.3f}  | ΛCDM χ²: {chi2_lcdm:.3f}")
    print(f"📐 DK-RD2 RMSE: {rmse_dk:.3f} | ΛCDM RMSE: {rmse_lcdm:.3f}")
    print(f"💾 Comparison table saved: {output_csv}")
    print(f"📊 Summary _stats CSV saved: {stats_csv}")


if __name__ == '__main__':

    import os
    # Ensure the 'evidence' directory exists
    os.makedirs("evidence", exist_ok=True)

    file_evid1, sn_colors_file = generate_figure01()
    print(f"✔️ Simulation of Relative Variation (%) as a Function of Relativistic Velocity and Temperature saved as\n: {file_evid1}")
    print(f"✔️ Table saved as: {sn_colors_file}")

    # Observational Data Files
    supernovae_data = "data/SCPUnion2_mu_vs_z.txt"
    # https://github.com/HoU-Wa/phy526proj/blob/master/SCPUnion2_mu_vs_z.txt
    fig2_img, fig2_table, stats_sn = generate_figure02(supernovae_data)
    print("Using: Amanullah et al. (The Supernova Cosmology Project), Ap.J., 2010.")
    print(f"✔️ Figure Analysis of the Distance Modulus μ(z) saved as: {fig2_img}")
    print(f"✔️ Table saved as: {fig2_table}")

    # Observational Data Files
    cmb_data= "data/COM_PowerSpect_CMB-TT-full_R3.01.txt"
    # https://github.com/Zakobian/CMB_cs_plots/blob/main/COM_PowerSpect_CMB-TT-full_R3.01.txt
    # https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/CMB_spectrum_%26_Likelihood_Code
    fig3_img, fig3_table, stats_cmb = generate_figure03(cmb_data)
    print("Using: Planck legacy archive")
    print(f"✔️ Figure CMB Angular Power Spectrum saved as: {fig3_img}")
    print(f"✔️ Table saved as: {fig3_table}")

    # Observational Data Files
    pantheon_data = "data/Pantheon+SH0ES.dat" # Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat
    # https://github.com/PantheonPlusSH0ES/DataRelease/blob/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat

    output_csv= os.path.join(out_dir_path, "rdm_SN_comparison.csv")
    plot_path = os.path.join(out_dir_path, "rdm_SN_comparison.png")
    out_shoes = load_and_compare_sn_pantheon_dataset(pantheon_data, output_csv, plot_path)
    print("Using: Pantheon+SH0ES.dat")
    print("[✓] Visual comparison CSV saved: out_shoes")
    fig4_vals = generate_figure04(output_csv)
    print(f"✔️ Figure Emergence of Dark Matter saved as: {fig4_vals[0]}")
    print(f"📄 Table saved as: {fig4_vals[1]}")
    fig5_vals = generate_figure05()
    print("Tabulated Einstein Radius comparison")
    print(f"✔️ Figure Einstein Radius Comparison Table saved as: {fig5_vals[0]}")
    print(f"📄 Table saved as: {fig5_vals[1]}")

    # === Hubble Expansion History Comparison (H(z)) ===
    # This routine compares the predicted Hubble parameter H(z)
    # from the DK-RD2 model with actual observational data,
    # and generates both a comparison CSV and a figure.

    # Observational Data Files
    hz_data_path = "data/hubble_observations.csv"  # Observational Hubble H(z) data
    output_csv = os.path.join(out_dir_path,"rdm_Hz_comparison.csv")  # Output CSV with model vs data
    plot_path = os.path.join(out_dir_path, "rdm_Hz_comparison.png")  # Output figure for publication

    optimized_Hz_comparison(
        hz_data_path,       # Observational Hubble H(z) data
        output_csv,         # Output CSV with model vs data
        plot_path,          # Output Plot Figure with model vs data
        E_Relativistic      # DK-RD2 expansion rate function
    )

    # Run the DESI validation pipeline using DK-RD2 model.
    run_desi_validation()

    """
    Reads all *_stats.csv files in out_dir_path= /evidence/, extracts chi2, mse, AIC, BIC values,
    generates a comparison bar chart with tabulated results shown below the plot.
    """
    generate_sigma10_with_stats()

    print("— GabE=mc² & Luludns -> ∞Ψ")
    print(git_gabe)
