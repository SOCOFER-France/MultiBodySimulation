import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import sys, socofer
sys.path.append(socofer.devpy_ala_path)


# Paramètres physiques
M = 1.0
k1 = 10.0
k2 = 15.0
c1 = 0.2
c2 = 0.32
x2_init = 0.1  # déplacement initial de M2

x1_up_limit = 0.02
x1_low_limit = -0.005
k_stop = 10000      # raideur de la butée
c_stop = 200        # amortissement local

t_span = [0, 30.0]
t_eval = np.linspace(t_span[0], t_span[1], 1000)
dt = t_eval[1] - t_eval[0]

fx0 = lambda t : np.sin(2*np.pi*t)*0.005 * (t<10) * 0.
v0 = lambda t : 2*np.pi*np.cos(2*np.pi*t)*0.005 * (t<10) * 0.

# Solution par scipy
def double_oscillator_with_stop(t, y):
    x1, v1, x2, v2 = y

    # Force ressort-amortisseur classiques
    F_spring1 = k1 * (fx0(t) - x1) + c1 * (v0(t) - v1)
    F_spring2 = k2 * (x2 - x1) + c2 * (v2 - v1)

    # Force de butée unilatérale (active uniquement quand x1 > x1_limit)
    F_stop = 0.0
    if x1 > fx0(t) + x1_up_limit :
        F_stop = k_stop * (fx0(t) + x1_up_limit - x1 ) + c_stop *(v0(t) - v1 )
    if x1 < fx0(t) + x1_low_limit:
        F_stop = k_stop * (fx0(t) + x1_low_limit - x1) + c_stop *(v0(t) - v1 )

    # Équations du mouvement
    a1 = (F_spring1 + F_spring2 + F_stop) / M
    a2 = (-F_spring2) / M

    return [v1, a1, v2, a2]

y0 = [0.0, 0.0, x2_init, 0.0]

sol = solve_ivp(double_oscillator_with_stop,
                t_span,
                y0,
                t_eval=t_eval,
                method="BDF")
x1_scipy, x2_scipy = sol.y[0], sol.y[2]
t_scipy = sol.t


from MultiBodySimulation.MBSBody import MBSRigidBody3D,MBSReferenceBody3D
from MultiBodySimulation.MBSMechanicalJoint import (MBSLinkLinearSpringDamper,
                                                MBSLinkHardStop)
from MultiBodySimulation.MBSMechanicalSystem import MBSMechanicalSystem3D


# Création du système
mecha_sys = MBSMechanicalSystem3D()
mecha_sys.gravity = np.array([0., 0., 0.])  # pas de gravité

# Corps
refBody = MBSReferenceBody3D("Sol")
M1 = MBSRigidBody3D("M1", mass=M, inertia_tensor=1)
M2 = MBSRigidBody3D("M2", mass=M, inertia_tensor=1)

# Position initiale
x0, x1, x2 = 0.0, 0.01, 0.02
ref_pos = [x0, 0., 0.]
M1_pos = [x1, 0., 0.]
M2_pos = [x2, 0., 0.]
M2_init_pos = np.array([x2+x2_init, 0.,0.]) # déplacée

refBody.SetDisplacementFunction(dx_func=fx0)

# Définition des positions initiales
refBody.SetReferencePosition(ref_pos)
M1.SetReferencePosition(M1_pos)
M2.SetReferencePosition(M2_pos)
M2.ChangeInitialPosition(M2_init_pos)

# Liaisons ressorts entre :
# - Sol et M1
joint1 = MBSLinkLinearSpringDamper(refBody,
                             ref_pos,
                             M1,
                             M1_pos,
                             stiffness = [k1, 0, 0],
                             damping = [c1, 0, 0])

hardstop = MBSLinkHardStop(refBody,
                             ref_pos,
                             M1,
                             M1_pos,
                             Tx_gap = (x1_up_limit, x1_low_limit),
                             penetration_tolerance=1e-4)

# - M1 et M2
joint2 = MBSLinkLinearSpringDamper(M1,
                             M1_pos,
                             M2,
                             M2_pos,
                             stiffness = [k2, 0, 0],
                             damping = [c2, 0, 0])



# Ajout au système
mecha_sys.AddRigidBody(refBody)
mecha_sys.AddRigidBody(M1)
mecha_sys.AddRigidBody(M2)
mecha_sys.AddLinkage(joint1)
mecha_sys.AddLinkage(joint2)
mecha_sys.AddLinkage(hardstop)

# Simulation
import time
t1 = time.time()
t_eval, results = mecha_sys.RunDynamicSimulation(t_span=t_span, dt=dt, ode_method="BDF")
t2 = time.time()
print(f"Elapsed time : {(t2-t1):.2f}")
x1 = results["M1"].displacements[0]
x2 = results["M2"].displacements[0]



# Affichage
plt.figure(figsize=(11, 6))

plt.subplot(1, 2, 1)
plt.plot(t_eval, x2, label="M2")
plt.plot(t_scipy, x2_scipy, label="M2 scipy", ls="--")
plt.xlabel("Temps [s]")
plt.ylabel("Position X [m]")
plt.title("Double oscillateur couplé")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(t_eval, x1, label="M1")
plt.plot(t_scipy, x1_scipy, label="M1 scipy", ls="--")
plt.xlabel("Temps [s]")
plt.ylabel("Position X [m]")
plt.title("Double oscillateur couplé")
plt.legend()
plt.grid()



# plt.subplot(1, 2, 2)
# plt.plot(x1, x2)
# plt.plot(x1_scipy, x2_scipy, ls="--")
# plt.xlabel("M1 [m]")
# plt.ylabel("M2 [m]")
# plt.title("Trajectoire couplée dans l’espace d’état")
# plt.grid()

plt.tight_layout()
plt.show()