

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import sys, socofer
sys.path.append(socofer.devpy_ala_path)

from MultiBodySimulation.MBSBody import MBSRigidBody3D,MBSReferenceBody3D
from MultiBodySimulation.MBSMechanicalJoint import MBSLinkLinearSpringDamper
from MultiBodySimulation.MBSMechanicalSystem import MBSLinearSystem

M = 1.0
k = 20
c = 1.5

zref = 0.0
zinit = 0.1
t_span = [0,8]
t_eval = np.linspace(t_span[0], t_span[1], 500)
dt = t_eval[1]-t_span[0]

# Solving with scipy
Amatrix = np.array([[0,1],
           [-k/M,-c/M]])
bvec = np.array([0.,-9.81])

r = solve_ivp(lambda t,z : Amatrix.dot(z) + bvec,
               t_span,
               [zinit,0.],
               t_eval=t_eval )
z_scipy = r.y[0]


# Solving multibody tools

# Création du système
mecha_sys = MBSLinearSystem()
mecha_sys.gravity = np.array([0,0,-9.81])


refBody = MBSReferenceBody3D("Reference")
massBody = MBSRigidBody3D("Mass",M,0.1)

ref_position = np.array([0,0,zref])

# Placer les corps dans le repère global
refBody.SetReferencePosition(ref_position)
massBody.SetReferencePosition(ref_position)

joint = MBSLinkLinearSpringDamper(refBody,
            ref_position, #Point d'attache au corps 1 dans le repère global
            massBody,
            ref_position, #Point d'attache au corps 2 dans le repère global
            k,
            c,
                            )

mecha_sys.AddRigidBody(refBody)
mecha_sys.AddRigidBody(massBody)
mecha_sys.AddLinkage(joint)

# On repositionne le corps dans le repère global pour initialiser la simulation
massBody.ChangeInitialPosition(np.array([0,0,zinit]))

#Simulation
t_mbs, results = mecha_sys.RunDynamicSimulation(t_span=t_span,dt=dt)
z_result = results["Mass"].positions[2]

plt.figure()
plt.plot(t_eval, z_scipy, label = "Scipy", color = "blue", lw=2)
plt.plot(t_mbs, z_result, ls="--", label = "MultiBody", color = "darkorange", lw=2)
plt.grid(True)
plt.xlabel('time (s)')
plt.ylabel("Position [m]")
plt.title("Comparaison : masse ressort simple")
plt.legend()

plt.show()