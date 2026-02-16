import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import sys
import socofer
sys.path.append(socofer.devpy_ala_path)

from Vehicle_essieux_parametrage import (mecha_sys,
                                         excitation_roue_11,excitation_roue_12,
                                         excitation_roue_21, excitation_roue_22,
                                         boite_11)

from vibrationSignalPSD import psd2time, compute_PSD, build_PSD_61373_amplitudes

simu_duration = 1.0 # s
f1 = 10
fstart = 20
fend = 100
f2 = 200
asd_amp = 8.74




psd_func = build_PSD_61373_amplitudes(asd_amp, fstart, fend)
spec = np.array([[f1,  psd_func(f1)],
                [fstart,  psd_func(fstart)],
                [fend,  psd_func(fend)],
                [f2,  psd_func(f2)]])

# Excitations
def generate_61373_sig():


    we = dict(portion=0.1)

    cond = False
    while not cond :
        acc_array,dt_freq,time_array = psd2time(spec,f1,f2,
                                df=1/simu_duration,
                                winends=we)
        vit_array = np.cumsum(acc_array * dt_freq)
        dep_array = np.cumsum(vit_array * dt_freq)

        cond = abs(dep_array.mean()) < 0.1 * np.abs(dep_array).mean()
    func = interp1d(time_array+0.01,
                    dep_array,
                    kind="linear",
                    bounds_error=False,
                    fill_value=0.0)
    return func


dz_func_11 = generate_61373_sig()
dz_func_12 = generate_61373_sig()
dz_func_21 = generate_61373_sig()
dz_func_22 = generate_61373_sig()

excitation_roue_11.SetDisplacementFunction(dz_func = dz_func_11)
excitation_roue_12.SetDisplacementFunction(dz_func = dz_func_12)
excitation_roue_21.SetDisplacementFunction(dz_func = dz_func_21)
excitation_roue_22.SetDisplacementFunction(dz_func = dz_func_22)


t_eval, results = mecha_sys.RunDynamicSimulation(t_span=[0, simu_duration],
                                                 dt=1e-3,
                                )
# Extraction des résultats
caisse_results = results["Caisse"]
essieu_1_results = results["Essieu 1"]
essieu_2_results = results["Essieu 2"]
boite_11_results = results["Boite 11"]
boite_12_results = results["Boite 12"]
boite_21_results = results["Boite 21"]
boite_22_results = results["Boite 22"]
excitation_roue_11_results = results["excitation 11"]
excitation_roue_12_results = results["excitation 12"]
excitation_roue_21_results = results["excitation 21"]
excitation_roue_22_results = results["excitation 22"]

# =============================================================================
# VISUALISATIONS AMÉLIORÉES
# =============================================================================

# Configuration générale des graphiques
plt.style.use('seaborn-v0_8-darkgrid')
colors = {'excitation': '#FF6B6B', 'essieu': '#4ECDC4', 'boite': '#45B7D1', 'caisse': '#96CEB4'}

# -------------------------------------------------------------------------
## FIGURE 1: SIGNAUX D'EXCITATION
# -------------------------------------------------------------------------
fig1 = plt.figure(figsize=(10, 8))
fig1.suptitle('Signaux d\'Excitation des 4 Roues', fontsize=16, fontweight='bold')


excitations = {
    'Roue 11 (AV Gauche)': excitation_roue_11_results.accelerations[2],
    'Roue 12 (AV Droite)': excitation_roue_12_results.accelerations[2],
    'Roue 21 (AR Gauche)': excitation_roue_21_results.accelerations[2],
    'Roue 22 (AR Droite)': excitation_roue_22_results.accelerations[2]
}

ax1 = plt.subplot(2,1,1)
ax2 = plt.subplot(2,1,2)
for idx, (label, signal) in enumerate(excitations.items(), 1):
    # Temporel
    ax1.plot(t_eval, signal , linewidth=1.5)


    # Fréquentiel
    f, Pxx = compute_PSD(t_eval,signal, f1, f2)

    ax2.loglog(f, Pxx, color=colors['excitation'], linewidth=1.5)

ax2.plot(spec[:,0], spec[:,1], color="k", label="Spectre 61373")

ax1.set_ylabel('Accélération [m/s²]', fontsize=10)
ax1.set_title("Accélération", fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('Temps [s]', fontsize=10)

ax2.set_ylabel('PSD [m²/Hz]', fontsize=10)
ax2.set_title('Densité Spectrale de Puissance', fontsize=11)
ax2.grid(True, alpha=0.3, which='both')
ax2.set_xlabel('Fréquence [Hz]', fontsize=10)

plt.tight_layout()

# -------------------------------------------------------------------------
# FIGURE 2: RÉPONSE DE LA CAISSE (3 AXES)
# -------------------------------------------------------------------------
fig2 = plt.figure(figsize=(14, 8))
fig2.suptitle('Réponse Dynamique de la Caisse', fontsize=16, fontweight='bold')

axes_labels = ['X (Longitudinal)', 'Y (Latéral)', 'Z (Vertical)']
axes_units = ['mm', 'mm', 'mm']

for idx in range(3):
    # Temporel
    ax1 = plt.subplot(3, 3, 3 * idx + 1)
    ax1.plot(t_eval, caisse_results.positions[idx] * 1000,
             color=colors['caisse'], linewidth=1.5, label='Déplacement')
    ax1.set_ylabel(f'Déplacement {axes_units[idx]}', fontsize=10)
    ax1.set_title(f'{axes_labels[idx]} - Temporel', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    if idx == 2:
        ax1.set_xlabel('Temps [s]', fontsize=10)

    # Vitesse
    ax2 = plt.subplot(3, 3, 3 * idx + 2)
    velocity = np.gradient(caisse_results.positions[idx], t_eval) * 1000
    ax2.plot(t_eval, velocity, color='#FF8C42', linewidth=1.5)
    ax2.set_ylabel(f'Vitesse [mm/s]', fontsize=10)
    ax2.set_title(f'{axes_labels[idx]} - Vitesse', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    rms_vel = np.sqrt(np.mean(velocity ** 2))
    ax2.text(0.02, 0.98, f'RMS = {rms_vel:.2f} mm/s',
             transform=ax2.transAxes, fontsize=9,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    if idx == 2:
        ax2.set_xlabel('Temps [s]', fontsize=10)

    # Accélération
    ax3 = plt.subplot(3, 3, 3 * idx + 3)
    acceleration = np.gradient(velocity, t_eval) / 1000  # m/s²
    ax3.plot(t_eval, acceleration, color='#E74C3C', linewidth=1.5)
    ax3.set_ylabel(f'Accélération [m/s²]', fontsize=10)
    ax3.set_title(f'{axes_labels[idx]} - Accélération', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    rms_acc = np.sqrt(np.mean(acceleration ** 2))
    ax3.text(0.02, 0.98, f'RMS = {rms_acc:.2f} m/s²',
             transform=ax3.transAxes, fontsize=9,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    if idx == 2:
        ax3.set_xlabel('Temps [s]', fontsize=10)

plt.tight_layout()

# -------------------------------------------------------------------------
# FIGURE 3: COMPARAISON ESSIEU - BOITE (Essieu 1, Roue 11)
# -------------------------------------------------------------------------
fig3 = plt.figure(figsize=(14, 8))
fig3.suptitle('Liaison Cinématique Essieu 1 - Boîte 11', fontsize=16, fontweight='bold')

essieu_1_boite11_results = essieu_1_results.get_connected_point_motion(
    boite_11.GetReferencePosition(), approx_rotation=True)

axes_comp = ['X', 'Y', 'Z']
for idx in range(3):
    ax = plt.subplot(2, 3, idx + 1)
    ax.plot(t_eval, essieu_1_boite11_results.positions[idx] * 1000,
            color=colors['essieu'], linewidth=2, label='Essieu 1')
    ax.plot(t_eval, boite_11_results.positions[idx] * 1000,
            color=colors['boite'], linewidth=2, linestyle='--', label='Boîte 11')
    ax.set_ylabel(f'Position {axes_comp[idx]} [mm]', fontsize=10)
    ax.set_title(f'Axe {axes_comp[idx]} - Positions', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    if idx == 0:
        ax.set_xlabel('Temps [s]', fontsize=10)

    # Écart (erreur cinématique)
    ax2 = plt.subplot(2, 3, idx + 4)
    error = (essieu_1_boite11_results.positions[idx] - boite_11_results.positions[idx]) * 1e6
    ax2.plot(t_eval, error, color='#E74C3C', linewidth=1.5)
    ax2.set_ylabel(f'Écart [µm]', fontsize=10)
    ax2.set_title(f'Axe {axes_comp[idx]} - Écart Cinématique', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Temps [s]', fontsize=10)

    max_error = np.max(np.abs(error))
    ax2.text(0.02, 0.98, f'Max = {max_error:.2f} µm',
             transform=ax2.transAxes, fontsize=9,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='pink', alpha=0.5))

plt.tight_layout()

# -------------------------------------------------------------------------
# FIGURE 4: ANALYSE SPECTRALE DE LA RÉPONSE VERTICALE
# -------------------------------------------------------------------------
fig4 = plt.figure(figsize=(14, 6))
fig4.suptitle('Analyse Fréquentielle - Axe Vertical (Z)', fontsize=16, fontweight='bold')

# Calcul des spectres
f_exc, Pxx_exc = compute_PSD(t_eval, excitation_roue_11_results.accelerations[2],f1,f2)
f_caisse, Pxx_caisse = compute_PSD(t_eval, caisse_results.accelerations[2],f1,f2)

# Spectre d'excitation
ax1 = plt.subplot(2, 1, 1)
ax1.loglog(f_exc, Pxx_exc, color=colors['excitation'], linewidth=2, label='Excitation Roue 11')
# Spectre de réponse
ax1.loglog(f_caisse, Pxx_caisse, color=colors['caisse'], linewidth=2, label='Réponse Caisse')

ax1.set_xlabel('Fréquence [Hz]', fontsize=11)
ax1.set_ylabel('ASD [(m/s²)²/Hz]', fontsize=11)
ax1.grid(True, alpha=0.3, which='both')
ax1.legend(fontsize=9)



# Fonction de transfert (approximation)
ax3 = plt.subplot(2, 1, 2)
# Interpolation pour avoir les mêmes fréquences
f_common = f_exc
Pxx_caisse_interp = np.interp(f_common, f_caisse, Pxx_caisse)
FRF = np.sqrt(Pxx_caisse_interp / (Pxx_exc + 1e-20))  # Éviter division par zéro

ax3.loglog(f_common, FRF, color='#9B59B6', linewidth=2)
ax3.set_xlabel('Fréquence [Hz]', fontsize=11)
ax3.set_ylabel('Transmissibilité [-]', fontsize=11)
ax3.set_title('Fonction de Transfert Approx.', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, which='both')
ax3.axhline(1, color='gray', linestyle='--', alpha=0.5, label='Unité')
ax3.legend(fontsize=9)

plt.tight_layout()


print("\n=== SIMULATION TERMINÉE ===")
print(f"Durée simulée : {simu_duration} s")
print(f"Nombre de points : {len(t_eval)}")
print(
    f"\nAccélération RMS caisse (Z) : {np.sqrt(np.mean((np.gradient(np.gradient(caisse_results.positions[2] * 1000, t_eval), t_eval) / 1000) ** 2)):.3f} m/s²")
print(f"Cible RMS : {asd_amp:.2f} m/s²")

plt.show()