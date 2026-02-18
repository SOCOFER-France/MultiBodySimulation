

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
from Manivelles_serie_param import *

t_final = 25
f = 1.5
amp = 5 * np.pi / 180
theta0_sig = lambda t : amp * np.sin(2*np.pi * f * (1+0.2*np.cos(2*np.pi*0.3*f*t)) * t)
omega0_sig = lambda t : (theta0_sig(t+1e-6) - theta0_sig(t-1e-6))/(2e-6)
dt = 0.001  # pas de temps
n_steps = int(t_final / dt)

t_eval = np.linspace(0, t_final, n_steps)
# ============================================================
# FONCTION DE RÉSOLUTION THÉORIQUE
# ============================================================

def system_dynamics(t, y):
    """
    Équation du mouvement : M*θ_ddot + C*θ_dot + K*θ = 0
    État : y = [θ1, θ2, θ1_dot, θ2_dot]
    """
    theta = y[0:2]
    theta_dot = y[2:4]

    # M*θ_ddot = -C*θ_dot - K*θ
    theta_ddot = np.linalg.solve(M_theo, -C_theo @ theta_dot - K_theo @ theta)

    theta_ddot[0] += (kx1*L1_b1**2 * theta0_sig(t) + cx1*L1_b1**2 * omega0_sig(t))/Jz1

    return np.concatenate([theta_dot, theta_ddot])

# ============================================================
# CONDITIONS INITIALES
# ============================================================

# Conditions initiales : angles et vitesses angulaires
theta1_0 = 0.
theta2_0 = 0.

y0_theo = np.array([theta1_0, theta2_0, 0., 0.])

print("\n" + "="*60)
print("CONDITIONS INITIALES")
print("="*60)
print(f"θ1(0) = {theta1_0:.4f} rad = {np.degrees(theta1_0):.2f}°")
print(f"θ2(0) = {theta2_0:.4f} rad = {np.degrees(theta2_0):.2f}°")


# ============================================================
# SIMULATION THÉORIQUE
# ============================================================




print("\n" + "="*60)
print("SIMULATION THÉORIQUE")
print("="*60)
print(f"Durée de simulation : {t_final} s")
print("Résolution en cours...")

sol_theo = solve_ivp(system_dynamics, [0, t_final], y0_theo,
                     method='BDF', t_eval=t_eval)

theta1_theo = sol_theo.y[0]
theta2_theo = sol_theo.y[1]
omega1_theo = sol_theo.y[2]
omega2_theo = sol_theo.y[3]

print("Résolution théorique terminée !")


# ============================================================
# CONFIGURATION INITIALE MBS
# ============================================================

# Définir les angles initiaux (rotation autour de z)
Crank1.ChangeInitialAngle([0.0, 0.0, theta1_0])
Crank2.ChangeInitialAngle([0.0, 0.0, theta2_0])

RefBody1.SetRotationFunction(dtheta_z_func = theta0_sig)


# ============================================================
# SIMULATION MBS
# ============================================================


# Configuration de la simulation


print(f"Configuration : dt = {dt} s, {n_steps} pas de temps")
print("Simulation en cours...")

t1 = time.time()
t_mbs, results_mbs = mecha_sys.RunDynamicSimulation(
    t_span=[0, t_final],
    dt=dt,
    solver_type="constraint_stabilized",
    stabilization_method="Lagrangian",
    print_steps = 25,
)
t2 = time.time()
print("Elapsed time: ", t2-t1)


print("Simulation MBS terminée !")

# Extraction des résultats MBS
theta1_mbs = results_mbs['Manivelle_1'].angles[2]  # Rotation autour de z
theta2_mbs = results_mbs['Manivelle_2'].angles[2]


# ============================================================
# CALCUL DES ERREURS
# ============================================================

# Interpolation pour comparer aux mêmes instants
from scipy.interpolate import interp1d

interp_theta1_theo = interp1d(sol_theo.t, theta1_theo, kind='cubic')
interp_theta2_theo = interp1d(sol_theo.t, theta2_theo, kind='cubic')

theta1_theo_interp = interp_theta1_theo(t_mbs)
theta2_theo_interp = interp_theta2_theo(t_mbs)

error_theta1 = np.abs(theta1_mbs - theta1_theo_interp)
error_theta2 = np.abs(theta2_mbs - theta2_theo_interp)

print("\n" + "="*60)
print("ERREURS DE SIMULATION")
print("="*60)
print(f"Erreur max θ1 : {np.max(error_theta1)*1000:.3f} mrad")
print(f"Erreur RMS θ1 : {np.sqrt(np.mean(error_theta1**2))*1000:.3f} mrad")
print(f"Erreur max θ2 : {np.max(error_theta2)*1000:.3f} mrad")
print(f"Erreur RMS θ2 : {np.sqrt(np.mean(error_theta2**2))*1000:.3f} mrad")


# ============================================================
# VISUALISATION
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(9, 7))

# Colonne 1 : Manivelle 1
axes[0, 0].plot(sol_theo.t, np.degrees(theta1_theo), 'b-',
                label='Théorie', linewidth=2)
axes[0, 0].plot(t_mbs, np.degrees(theta1_mbs), 'r--',
                label='MBS', linewidth=1.5, alpha=0.8)
axes[0, 0].set_ylabel('θ1 [°]')
axes[0, 0].set_title('Angle Manivelle 1')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)


axes[1, 0].semilogy(t_mbs, error_theta1, 'g-', linewidth=1.5)
axes[1, 0].set_xlabel('Temps [s]')
axes[1, 0].set_ylabel('Erreur θ1 [rad]')
axes[1, 0].set_title('Erreur absolue Manivelle 1')
axes[1, 0].grid(True, alpha=0.3)

# Colonne 2 : Manivelle 2
axes[0, 1].plot(sol_theo.t, np.degrees(theta2_theo), 'b-',
                label='Théorie', linewidth=2)
axes[0, 1].plot(t_mbs, np.degrees(theta2_mbs), 'r--',
                label='MBS', linewidth=1.5, alpha=0.8)
axes[0, 1].set_ylabel('θ2 [°]')
axes[0, 1].set_title('Angle Manivelle 2')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 1].semilogy(t_mbs, error_theta2, 'g-', linewidth=1.5)
axes[1, 1].set_xlabel('Temps [s]')
axes[1, 1].set_ylabel('Erreur θ2 [rad]')
axes[1, 1].set_title('Erreur absolue Manivelle 2')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('crank_system_time_simulation.png', dpi=150)
print("\nGraphique sauvegardé : 'crank_system_time_simulation.png'")
plt.show()



print("\n" + "="*60)
print("SIMULATION TERMINÉE")
print("="*60)