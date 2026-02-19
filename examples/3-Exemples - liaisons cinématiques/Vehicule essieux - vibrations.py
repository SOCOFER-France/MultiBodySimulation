import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time
import sys
import socofer
sys.path.append(socofer.devpy_ala_path)

from Vehicle_essieux_parametrage import prepare_study

from vibrationSignalPSD import psd2time, compute_PSD, build_PSD_61373_amplitudes

simu_duration = 0.5 # s
f1 = 10
fstart = 20
fend = 100
f2 = 200
asd_amp = 8.74
dt = 5e-4

include_butee = True
e_butee = 2e-3

algo1 = "constraint_stabilized"
algo2 = "constraint_stabilized"

method1 = ""
method2 = "Lagrange"

label1 = "BDF2 - pénalisation"
label2 = "BDF2 - Lagrange"

tol1 = 1e-6
tol2 = 1000

adaptative = True



(mecha_sys,
 excitation_roue_11,excitation_roue_12,
 excitation_roue_21, excitation_roue_22,
 boite_11) = prepare_study(tol1,
                           include_butee = include_butee,
                           e_butee=e_butee,)

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

mecha_sys.ComputeQrDecomposedSystem()

t1 = time.time()
t_eval, results = mecha_sys.RunDynamicSimulation(t_span=[0, simu_duration],
                                                 dt=dt,
                                                 solver_type=algo1,
                                                 stabilization_method=method1,
                                                 adaptative = adaptative,
                                                 print_steps = 10,
                                                 print_inner_iter = False
                                                 )
t2 = time.time()
print("Elapsed time: ", t2-t1)

t1 = time.time()


(mecha_sys,
 excitation_roue_11,excitation_roue_12,
 excitation_roue_21, excitation_roue_22,
 boite_11) = prepare_study(tol2,
                           include_butee = include_butee,
                           e_butee=e_butee,)

excitation_roue_11.SetDisplacementFunction(dz_func = dz_func_11)
excitation_roue_12.SetDisplacementFunction(dz_func = dz_func_12)
excitation_roue_21.SetDisplacementFunction(dz_func = dz_func_21)
excitation_roue_22.SetDisplacementFunction(dz_func = dz_func_22)


t_eval_scipy, results_scipy = mecha_sys.RunDynamicSimulation(t_span=[0, simu_duration],
                                                 dt = dt,
                                                 solver_type = algo2,
                                                 stabilization_method = method2,
                                                 adaptative = adaptative,
                                                 print_steps = 10,
                                                 print_inner_iter = False
                                )
t2 = time.time()
print("Elapsed time: ", t2-t1)

# Extraction des résultats
caisse_results = results["Caisse"]
essieu_1_results = results["Essieu 1"]
boite_11_results = results["Boite 11"]


# Extraction des résultats
caisse_results_scipy = results_scipy["Caisse"]
essieu_1_results_scipy = results_scipy["Essieu 1"]
boite_11_results_scipy = results_scipy["Boite 11"]

# =============================================================================
# VISUALISATIONS AMÉLIORÉES
# =============================================================================

# Configuration générale des graphiques
plt.style.use('seaborn-v0_8-darkgrid')



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
             linewidth=1.0, label=f'{label1}')
    ax1.plot(t_eval_scipy, caisse_results_scipy.positions[idx] * 1000,
             linewidth=2.0, alpha = 0.5, label=f'{label2}')
    ax1.set_ylabel(f'Déplacement {axes_units[idx]}', fontsize=10)
    ax1.set_title(f'{axes_labels[idx]} - Temporel', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    if idx == 2:
        ax1.set_xlabel('Temps [s]', fontsize=10)

    # Vitesse
    ax2 = plt.subplot(3, 3, 3 * idx + 2)
    velocity =caisse_results.velocities[idx] * 1000
    velocity_scipy = caisse_results_scipy.velocities[idx] * 1000
    ax2.plot(t_eval, velocity, label=f'{label1}', linewidth=1)
    ax2.plot(t_eval_scipy, velocity_scipy, label=f'{label2}', linewidth=2, alpha = 0.5)
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
    acceleration = caisse_results.accelerations[idx] * 1000
    acceleration_scipy = caisse_results_scipy.accelerations[idx] * 1000
    ax3.plot(t_eval, acceleration, linewidth=1)
    ax3.plot(t_eval_scipy, acceleration_scipy, linewidth=2, alpha = 0.5)
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
            linewidth=2, label='Essieu 1')
    ax.plot(t_eval, boite_11_results.positions[idx] * 1000,
            linewidth=2, linestyle='--', label='Boîte 11')
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



fig4 = plt.figure(figsize=(14, 8))
fig4.suptitle('Liaison Cinématique Essieu 1 - Boîte 11 - SciPy', fontsize=16, fontweight='bold')

essieu_1_boite11_results_scipy = essieu_1_results_scipy.get_connected_point_motion(
    boite_11.GetReferencePosition(), approx_rotation=True)

axes_comp = ['X', 'Y', 'Z']
for idx in range(3):
    ax = plt.subplot(2, 3, idx + 1)
    ax.plot(t_eval_scipy, essieu_1_boite11_results_scipy.positions[idx] * 1000,
            linewidth=2, label='Essieu 1')
    ax.plot(t_eval_scipy, boite_11_results_scipy.positions[idx] * 1000,
            linewidth=2, linestyle='--', label='Boîte 11')
    ax.set_ylabel(f'Position {axes_comp[idx]} [mm]', fontsize=10)
    ax.set_title(f'Axe {axes_comp[idx]} - Positions', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    if idx == 0:
        ax.set_xlabel('Temps [s]', fontsize=10)

    # Écart (erreur cinématique)
    ax2 = plt.subplot(2, 3, idx + 4)
    error = (essieu_1_boite11_results_scipy.positions[idx] - boite_11_results_scipy.positions[idx]) * 1e6
    ax2.plot(t_eval_scipy, error, color='#E74C3C', linewidth=1.5)
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


plt.figure()

dz_boite11 = results["Boite 11"].positions[2] - results["Caisse"].get_connected_point_motion(boite_11.GetReferencePosition()).positions[2]

plt.subplot(211)
plt.plot(t_eval, dz_boite11, label = "Boite 11", linewidth=1.5)
plt.xlabel('Temps [s]', fontsize=10)
plt.ylabel('Position [mm]', fontsize=10)
plt.legend(fontsize=9)
plt.grid(True)

dz_boite11 = results_scipy["Boite 11"].positions[2] - results_scipy["Caisse"].get_connected_point_motion(boite_11.GetReferencePosition()).positions[2]


plt.subplot(212)
plt.title(label2)
plt.plot(t_eval_scipy, dz_boite11, label = "Boite 11", linewidth=1.5)
plt.xlabel('Temps [s]', fontsize=10)
plt.ylabel('Position [mm]', fontsize=10)
plt.legend(fontsize=9)
plt.grid(True)


plt.show()