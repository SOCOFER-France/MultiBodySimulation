import numpy as np
import matplotlib.pyplot as plt

import sys, socofer
sys.path.append(socofer.devpy_ala_path)

from MultiBodySimulation.MBSBody import MBSRigidBody3D,MBSReferenceBody3D
from MultiBodySimulation.MBSMechanicalJoint import (MBSLinkKinematic)
from MultiBodySimulation.MBSMechanicalSystem import MBSMechanicalSystem3D



from scipy.integrate import solve_ivp
# parametres physiques

m = 1
r = 0.2
Jm = 0.1
g = 9.81
Ieff = Jm + m * r**2 # Inertie au point de pivot
theta_0 = np.deg2rad(1)
rot0 = np.array([[1,    0          ,        0       ],
                 [0,np.cos(theta_0),-np.sin(theta_0)],
                 [0,np.sin(theta_0),np.cos(theta_0)]])
t_end = 10.0
dt = 0.01
t_eval = np.arange(0, t_end, dt)

# --- Système mécanique ---
sys = MBSMechanicalSystem3D()
sys.gravity = np.array([0, 0, -g])

# Corps 1 : Référence
body1 = MBSReferenceBody3D("ref")
sys.AddRigidBody(body1)

# Corps 2 : Masse pendulaire
body2 = MBSRigidBody3D("mass", mass=m, inertia_tensor=Jm)
body2.SetReferencePosition(np.array([0.0, 0, -r]))
sys.AddRigidBody(body2)

# Liaison pivot (bloque tout sauf rotation autour de X)
pivot = MBSLinkKinematic(body1,
                        body1.GetReferencePosition(),
                        body2,
                        body1.GetReferencePosition(),
                        Tx=1, Ty=1, Tz=1,
                        Ry=1, Rz=1)
sys.AddLinkage(pivot)

# Appliquer la rotation initiale autour de X

pos_init = rot0 @ body2.GetReferencePosition()
body2.ChangeInitialPosition(pos_init)
body2.ChangeInitialAngle([theta_0, 0, 0])

# Simulation multibody
t_result, results = sys.RunDynamicSimulation(t_span=[0, t_end], dt=dt)
positions = results["mass"].positions
y_pos = positions[1]
z_pos = positions[2]

# --- Résolution analytique avec scipy (approx. petit angle) ---
def pendulum_rhs(t, y):
    return [y[1], -m * g * r / Ieff * np.sin(y[0])]

# Le pendule est un corps solide donc au lieu d'écrire
# l'équation d²theta/dt² = -g/r * sin( theta)
# on a l'équation
# Ieff = d²theta/dt² = -g*m*r * sin( theta)

sol = solve_ivp(pendulum_rhs,
                [0, t_end],
                [theta_0, 0],
                t_eval=t_eval)
t_scipy = sol.t
theta_ref = sol.y[0]
y_ref = r * np.sin(theta_ref)
z_ref = -r * np.cos(theta_ref)

# --- Affichage graphique ---
fig, axs = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)

# 1. Mouvement dans le temps
axs[0].plot(t_result, y_pos, label="Y (simulation)", color='tab:blue')
axs[0].plot(t_result, z_pos, label="Z (simulation)", color='tab:orange')
axs[0].plot(t_scipy, y_ref, label="Y (réf. scipy)", color='red', ls = '--')
axs[0].plot(t_scipy, z_ref, label="Z (réf. scipy)", color='blue', ls="--")
axs[0].set_title("Mouvement du pendule dans le temps")
axs[0].set_xlabel("Temps [s]")
axs[0].set_ylabel("Position [m]")
axs[0].legend()
axs[0].grid(True)

# 2. Trajectoire dans le plan YZ
axs[1].plot(y_pos, z_pos, label="Simulation", color="tab:purple")
axs[1].plot(y_ref, z_ref, label="Réf. scipy", color="tab:gray",ls = '--')
axs[1].set_title("Trajectoire du pendule dans le plan YZ")
axs[1].set_xlabel("Y [m]")
axs[1].set_ylabel("Z [m]")
axs[1].axis("equal")
axs[1].legend()
axs[1].grid(True)


fig = plt.figure()

plt.plot(t_scipy, theta_ref)
plt.plot(t_result, results["mass"].angles[0])

plt.show()