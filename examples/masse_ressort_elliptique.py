import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import sys, socofer
sys.path.append(socofer.devpy_ala_path)

from MultiBodySimulation.MBSBody import MBSRigidBody3D,MBSReferenceBody3D
from MultiBodySimulation.MBSMechanicalJoint import MBSLinkLinearSpringDamper
from MultiBodySimulation.MBSMechanicalSystem import MBSMechanicalSystem3D

# Paramètres physiques
M = 1.0
k = 10.0      # raideur ressort
c = 0.3       # amortissement
x_init = 0.2  # position initiale
z_init = 0.1

t_span = [0, 15]
t_eval = np.linspace(t_span[0], t_span[1], 1000)
dt = t_eval[1] - t_eval[0]

kx = 1.3*k
cx = 1.3*c
kz = 0.7*k
cz = 0.7*c

# Resolution avec scipy
def oscillator_2D(t, y):
    x, vx, z, vz = y
    ax = (-cx * vx - kx * x) / M
    az = (-cz * vz - kz * z) / M
    return [vx, ax, vz, az]


# Conditions initiales
y0 = [x_init, 0.0, z_init, 0.0]

# Résolution
sol = solve_ivp(oscillator_2D, t_span, y0, t_eval=t_eval, method='RK45')
t_scipy = sol.t
x_scipy = sol.y[0]
z_scipy = sol.y[2]

# Création du système
mecha_sys = MBSMechanicalSystem3D()
mecha_sys.gravity = np.array([0., 0., 0.])  # pas de gravité

# Corps fixes et mobiles
refBody = MBSReferenceBody3D("Reference")
massBody = MBSRigidBody3D("Mass", mass=M, inertia_tensor=0.1)

ref_position = np.array([0., 0., 0.])
ref_angle = np.zeros(3)

# Définir les positions initiales
refBody.SetReferencePosition(ref_position)
massBody.SetReferencePosition(ref_position)

# Liaisons ressorts dans X et Z
joint_x = MBSLinkLinearSpringDamper(refBody,
                              ref_position,
                              massBody,
                              ref_position,
                              [kx,0,0],
                              [cx,0,0],)  # ressort en X

joint_z = MBSLinkLinearSpringDamper(refBody,
                              ref_position,
                              massBody,
                              ref_position,
                              [0,0,kz],
                              [0,0,cz],)  # ressort en Z

# Ajouter les éléments au système
mecha_sys.AddRigidBody(refBody)
mecha_sys.AddRigidBody(massBody)
mecha_sys.AddLinkage(joint_x)
mecha_sys.AddLinkage(joint_z)

# Position initiale
massBody.ChangeInitialPosition(np.array([x_init, 0., z_init]))

# Simulation
t_mbs, results = mecha_sys.RunDynamicSimulation(t_span=t_span, dt=dt)
x_result = results["Mass"].positions[0]
z_result = results["Mass"].positions[2]

# Affichage
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)

plt.plot(t_eval, x_result, label='x(t)')
plt.plot(t_eval, z_result, label='z(t)')
plt.plot(t_scipy, x_scipy, label='x(t) scipy', ls = "--")
plt.plot(t_scipy, z_scipy, label='z(t) scipy', ls = "--")
plt.xlabel("Temps [s]")
plt.ylabel("Position [m]")
plt.title("Composantes X et Z dans le temps")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x_result, z_result)
plt.plot(x_scipy, z_scipy, ls = "--")
plt.xlabel("x [m]")
plt.ylabel("z [m]")
plt.title("Trajectoire dans le plan XZ (oscillation elliptique)")
plt.grid(True)

plt.tight_layout()
plt.show()