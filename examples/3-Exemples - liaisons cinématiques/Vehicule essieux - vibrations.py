import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import butter, filtfilt, welch
from scipy.interpolate import interp1d

import sys
import socofer
sys.path.append(socofer.devpy_ala_path)

from MultiBodySimulation.MBSBody import MBSRigidBody3D,MBSReferenceBody3D
from MultiBodySimulation.MBSMechanicalJoint import MBSLinkLinearSpringDamper, MBSLinkKinematic
from MultiBodySimulation.MBSMechanicalSystem import MBSLinearSystem

from vibrationSignalPSD import psd2time, compute_PSD, build_PSD_61373_amplitudes

simu_duration = 1.0 # s
f1 = 10
fstart = 20
fend = 100
f2 = 200
asd_amp = 8.74


kinematic_tolerance = 1e-8


Lx_loco = 16.0 # Longueur totale
Hz_loco = 1.0
Hy_loco = 2.0
Lx_loco_essieu = 10.0 # Longueur entre essieux

L_essieu = 1.5 # Longueur essieu
L_boite = 1000e-3 # longueur de boite
H_boite = 200e-3 # hauteur et largeur de boite
Ly_roue = 1435e-3

K_spring = 1200e3 # N/m
C_spring = K_spring * 0.01 #1%
K_lat_spring = 1500
C_lat_spring = K_lat_spring * 0.01

rho = 7800 # densité acier

De = 150e-3 # Diametre essieu
Dr = 920e-3 # Diametre roue
Lr = 150e-3

Me = rho * L_essieu * np.pi * De**2/4
Mr = rho * Lr * np.pi * Dr**2/4

Jy_ess = 1/2 * Me * De**2/4
Jy_roue = 1/2 * Mr * Dr**2/4

Jx_ess = 1/4 * Me * De**2/4 + 1/12 * Me * L_essieu**2
Jx_roue = 1/4 * Mr * Dr**2/4 + 1/12 * Mr * Lr**2

Mb = L_boite * H_boite**2 * rho
Jxb = Mb/12 * (2*H_boite**2)
Jyb = Mb/12 * (L_boite**2 + H_boite**2)
Jzb = Mb/12 * (L_boite**2 + H_boite**2)

Munsuspended = Mb * 4 + Me * 2 + Mr * 4
Mloc = 30e3 - Munsuspended
Jx_loc = Mloc/12 * (Hy_loco**2 + Hz_loco**2)
Jy_loc = Mloc/12 * (Lx_loco**2 + Hz_loco**2)
Jz_loc = Mloc/12 * (Hy_loco**2 + Lx_loco**2)

print(f"Masse non suspendue : { Munsuspended:.2f} kg")
print(f"Masse caisse : {Mloc :.2f} kg")

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

# Solides

caisse_loco = MBSRigidBody3D("Caisse", Mloc, [Jx_loc, Jy_loc, Jz_loc])
caisse_loco.SetReferencePosition([0., 0., 2.2])

essieu_1 = MBSRigidBody3D("Essieu 1", Me + 2*Mr,
                          [Jx_ess+2*Jx_roue,
                           Jy_ess+2*Jy_roue,
                           Jx_ess+2*Jx_roue]) # Jx = Jz
essieu_2 = MBSRigidBody3D("Essieu 2", Me + 2*Mr,
                          [Jx_ess+2*Jx_roue,
                           Jy_ess+2*Jy_roue,
                           Jx_ess+2*Jx_roue]) # Jx = Jz
essieu_1.SetReferencePosition([-Lx_loco_essieu/2, 0., Dr/2])
essieu_2.SetReferencePosition([ Lx_loco_essieu/2, 0., Dr/2])

boite_11 = MBSRigidBody3D("Boite 11", Mb, [Jxb, Jyb, Jzb])
boite_12 = MBSRigidBody3D("Boite 12", Mb, [Jxb, Jyb, Jzb])
boite_21 = MBSRigidBody3D("Boite 21", Mb, [Jxb, Jyb, Jzb])
boite_22 = MBSRigidBody3D("Boite 22", Mb, [Jxb, Jyb, Jzb])

boite_11.SetReferencePosition([-Lx_loco_essieu/2, -L_essieu/2, Dr/2])
boite_12.SetReferencePosition([-Lx_loco_essieu/2,  L_essieu/2, Dr/2])
boite_21.SetReferencePosition([ Lx_loco_essieu/2, -L_essieu/2, Dr/2])
boite_22.SetReferencePosition([ Lx_loco_essieu/2,  L_essieu/2, Dr/2])


excitation_roue_11 = MBSReferenceBody3D("excitation 11")
excitation_roue_12 = MBSReferenceBody3D("excitation 12")
excitation_roue_21 = MBSReferenceBody3D("excitation 21")
excitation_roue_22 = MBSReferenceBody3D("excitation 22")

excitation_roue_11.SetReferencePosition([-Lx_loco_essieu/2, -Ly_roue/2, Dr/2])
excitation_roue_12.SetReferencePosition([-Lx_loco_essieu/2,  Ly_roue/2, Dr/2])
excitation_roue_21.SetReferencePosition([ Lx_loco_essieu/2, -Ly_roue/2, Dr/2])
excitation_roue_22.SetReferencePosition([ Lx_loco_essieu/2,  Ly_roue/2, Dr/2])

dz_func_11 = generate_61373_sig()
dz_func_12 = generate_61373_sig()
dz_func_21 = generate_61373_sig()
dz_func_22 = generate_61373_sig()

excitation_roue_11.SetDisplacementFunction(dz_func=dz_func_11)
excitation_roue_12.SetDisplacementFunction(dz_func=dz_func_12)
excitation_roue_21.SetDisplacementFunction(dz_func=dz_func_21)
excitation_roue_22.SetDisplacementFunction(dz_func=dz_func_22)

# Liaisons
pivot_boite_11 = MBSLinkKinematic(boite_11,
                               boite_11.GetReferencePosition(),
                               essieu_1,
                               boite_11.GetReferencePosition(),
                                Tx = 1,Rx = 1,
                                Ty = 1,Ry = 0,
                                Tz = 1,Rz = 1,
                                  kinematic_tolerance = kinematic_tolerance)
pivot_boite_12 = MBSLinkKinematic(boite_12,
                               boite_12.GetReferencePosition(),
                               essieu_1,
                               boite_12.GetReferencePosition(),
                                Tx = 1,Rx = 1,
                                Ty = 1,Ry = 0,
                                Tz = 1,Rz = 1,
                                  kinematic_tolerance = kinematic_tolerance)
pivot_boite_21 = MBSLinkKinematic(boite_21,
                               boite_21.GetReferencePosition(),
                               essieu_2,
                               boite_21.GetReferencePosition(),
                                Tx = 1,Rx = 1,
                                Ty = 1,Ry = 0,
                                Tz = 1,Rz = 1,
                                  kinematic_tolerance = kinematic_tolerance)
pivot_boite_22 = MBSLinkKinematic(boite_22,
                               boite_22.GetReferencePosition(),
                               essieu_2,
                               boite_22.GetReferencePosition(),
                                Tx = 1,Rx = 1,
                                Ty = 1,Ry = 0,
                                Tz = 1,Rz = 1,
                                  kinematic_tolerance = kinematic_tolerance)


fixation_roue_11 = MBSLinkKinematic(excitation_roue_11,
                               excitation_roue_11.GetReferencePosition(),
                               essieu_1,
                               excitation_roue_11.GetReferencePosition(),
                                Tz = 1,
                                  kinematic_tolerance = kinematic_tolerance)
fixation_roue_12 = MBSLinkKinematic(excitation_roue_12,
                               excitation_roue_12.GetReferencePosition(),
                               essieu_1,
                               excitation_roue_12.GetReferencePosition(),
                                Tz = 1,
                                  kinematic_tolerance = kinematic_tolerance)
fixation_roue_21 = MBSLinkKinematic(excitation_roue_21,
                               excitation_roue_21.GetReferencePosition(),
                               essieu_2,
                               excitation_roue_21.GetReferencePosition(),
                                Tz = 1,
                                  kinematic_tolerance = kinematic_tolerance)
fixation_roue_22 = MBSLinkKinematic(excitation_roue_22,
                               excitation_roue_22.GetReferencePosition(),
                               essieu_2,
                               excitation_roue_22.GetReferencePosition(),
                                Tz = 1,
                                  kinematic_tolerance = kinematic_tolerance)



# Ressorts
Kmat = [K_lat_spring, K_lat_spring, K_spring]
Cmat = [C_lat_spring, C_lat_spring, C_spring]
delta_x_boite =  np.array([L_boite/2,0,0])

ressort_1_boite_11 = MBSLinkLinearSpringDamper(boite_11,
                       boite_11.GetReferencePosition() + delta_x_boite,
                               caisse_loco,
                       boite_11.GetReferencePosition() + delta_x_boite,
                        stiffness = Kmat,
                        damping = Cmat)
ressort_2_boite_11 = MBSLinkLinearSpringDamper(boite_11,
                       boite_11.GetReferencePosition() - delta_x_boite,
                               caisse_loco,
                       boite_11.GetReferencePosition() - delta_x_boite,
                        stiffness = Kmat,
                        damping = Cmat)

ressort_1_boite_12 = MBSLinkLinearSpringDamper(boite_12,
                       boite_12.GetReferencePosition() + delta_x_boite,
                               caisse_loco,
                       boite_12.GetReferencePosition() + delta_x_boite,
                        stiffness = Kmat,
                        damping = Cmat)
ressort_2_boite_12 = MBSLinkLinearSpringDamper(boite_12,
                       boite_12.GetReferencePosition() - delta_x_boite,
                               caisse_loco,
                       boite_12.GetReferencePosition() - delta_x_boite,
                        stiffness = Kmat,
                        damping = Cmat)

ressort_1_boite_21 = MBSLinkLinearSpringDamper(boite_21,
                       boite_21.GetReferencePosition() + delta_x_boite,
                               caisse_loco,
                       boite_21.GetReferencePosition() + delta_x_boite,
                        stiffness = Kmat,
                        damping = Cmat)
ressort_2_boite_21 = MBSLinkLinearSpringDamper(boite_21,
                       boite_21.GetReferencePosition() - delta_x_boite,
                               caisse_loco,
                       boite_21.GetReferencePosition() - delta_x_boite,
                        stiffness = Kmat,
                        damping = Cmat)

ressort_1_boite_22 = MBSLinkLinearSpringDamper(boite_22,
                       boite_22.GetReferencePosition() + delta_x_boite,
                               caisse_loco,
                       boite_22.GetReferencePosition() + delta_x_boite,
                        stiffness = Kmat,
                        damping = Cmat)
ressort_2_boite_22 = MBSLinkLinearSpringDamper(boite_22,
                       boite_22.GetReferencePosition() - delta_x_boite,
                               caisse_loco,
                       boite_22.GetReferencePosition() - delta_x_boite,
                        stiffness = Kmat,
                        damping = Cmat)


# Assemblage
mecha_sys = MBSLinearSystem()
mecha_sys.AddRigidBody(caisse_loco)
mecha_sys.AddRigidBody( essieu_1 )
mecha_sys.AddRigidBody( essieu_2 )
mecha_sys.AddRigidBody( boite_11 )
mecha_sys.AddRigidBody( boite_12 )
mecha_sys.AddRigidBody( boite_21 )
mecha_sys.AddRigidBody( boite_22 )
mecha_sys.AddRigidBody( excitation_roue_11 )
mecha_sys.AddRigidBody( excitation_roue_12 )
mecha_sys.AddRigidBody( excitation_roue_21 )
mecha_sys.AddRigidBody( excitation_roue_22 )

mecha_sys.AddLinkage( pivot_boite_11 )
mecha_sys.AddLinkage( pivot_boite_12 )
mecha_sys.AddLinkage( pivot_boite_21 )
mecha_sys.AddLinkage( pivot_boite_22 )
mecha_sys.AddLinkage( fixation_roue_11 )
mecha_sys.AddLinkage( fixation_roue_12 )
mecha_sys.AddLinkage( fixation_roue_21 )
mecha_sys.AddLinkage( fixation_roue_22 )

mecha_sys.AddLinkage( ressort_1_boite_11 )
mecha_sys.AddLinkage( ressort_2_boite_11 )
mecha_sys.AddLinkage( ressort_1_boite_12 )
mecha_sys.AddLinkage( ressort_2_boite_12 )
mecha_sys.AddLinkage( ressort_1_boite_21 )
mecha_sys.AddLinkage( ressort_2_boite_21 )
mecha_sys.AddLinkage( ressort_1_boite_22 )
mecha_sys.AddLinkage( ressort_2_boite_22 )

mecha_sys.CheckUnconstrainedDegreeOfFreedom()

t_eval, results = mecha_sys.RunDynamicSimulation(t_span=[0, simu_duration],
                                                 dt=1e-3,
                                                 print_step_rate=0,
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
ax1.set_title(label, fontsize=11, fontweight='bold')
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