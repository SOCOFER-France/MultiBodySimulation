import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

J1 = 1.0
J2 = 4.0
J3 = 0.5
kt1 = 10.0
kt2 = 10.0
kt3 = 4.0
ct1 = kt1 * 8/100
ct2 = kt2 * 8/100
ct3 = kt3 * 8/100

t_span = [0, 30]
t_eval = np.linspace(t_span[0], t_span[1], 1000)
dt = t_eval[1] - t_eval[0]

# Positions
x1 = 0.01
x2 = x1 + 0.01
x3 = x2 + 0.01

# Torsion initiale
theta_3_init = 20*np.pi/180
y0_scipy = np.array([0.0, 0.0, theta_3_init, 0.0, 0.0, 0.0])

f = 0.5032 # Fréquence de raisonnance de J1;k1
theta0_sig = lambda t : np.sin(2*np.pi * f * t) * 20*np.pi/180 * 0
omega0_sig = lambda t : 2*np.pi*f*np.cos(2*np.pi*f*t) * 20*np.pi/180 * 0

## Solution analytique
M = np.diag([0,J1,J2,J3])
K = np.array([[kt1,     -kt1,    0,   0],
              [-kt1,  kt1+kt2,    -kt2,   0],
              [0,    -kt2,    kt2+kt3,   -kt3],
              [0,    0,    -kt3,   kt3]])
C = np.array([[ct1,     -ct1,    0,   0],
              [-ct1,  ct1+ct2,    -ct2,   0],
              [0,    -ct2,    ct2+ct3,   -ct3],
              [0,    0,    -ct3,   ct3]])

fixeddof = [0]
freedof = [1,2,3]

Mf = M[freedof][:,freedof]
Cf = C[freedof][:,freedof]
Kf = K[freedof][:,freedof]

Cb = C[freedof][:,fixeddof]
Kb = K[freedof][:,fixeddof]

invMf = np.linalg.inv(Mf)
def torsional_vibration_equation(t, y):
    theta = y[:3]
    omega = y[3:]
    gamma = -invMf @ (Kf @ theta + Cf @ omega + Kb @ [theta0_sig(t)] + Cb @ [omega0_sig(t)])

    return np.concatenate([omega, gamma])


sol = solve_ivp(torsional_vibration_equation,
                t_span,
                y0_scipy,
                t_eval=t_eval,
                method="BDF")
theta1_scipy, theta2_scipy, theta3_scipy = sol.y[0], sol.y[1], sol.y[2]
t_scipy = sol.t

## Etude avec MultibodySimulation
import sys,socofer
sys.path.append(socofer.devpy_ala_path)
from MultiBodySimulation.MBSBody import MBSRigidBody3D,MBSReferenceBody3D
from MultiBodySimulation.MBSMechanicalJoint import MBSLinkLinearSpringDamper
from MultiBodySimulation.MBSMechanicalSystem import MBSLinearSystem

mecha_sys = MBSLinearSystem()
mecha_sys.gravity = np.array([0.0,0.0,0.0])

RefBody = MBSReferenceBody3D("Ref")
RefBody.SetReferencePosition([0., 0., 0.])
RefBody.SetRotationFunction(dtheta_x_func = theta0_sig)

Mass1 = MBSRigidBody3D("Masse 1",
                        mass = 1,
                        inertia_tensor = J1)
Mass1.SetReferencePosition([x1, 0., 0.])

Mass2 = MBSRigidBody3D("Masse 2",
                        mass = 1,
                        inertia_tensor = J2)
Mass2.SetReferencePosition([x2, 0., 0.])

Mass3 = MBSRigidBody3D("Masse 3",
                        mass = 1,
                        inertia_tensor = J3)
Mass3.SetReferencePosition([x3, 0., 0.])

Mass3.ChangeInitialAngle([theta_3_init, 0., 0.])

mecha_sys.AddRigidBody(RefBody)
mecha_sys.AddRigidBody(Mass1)
mecha_sys.AddRigidBody(Mass2)
mecha_sys.AddRigidBody(Mass3)

joint01 = MBSLinkLinearSpringDamper(RefBody,
                                    [(x1)/2, 0., 0.],
                                    Mass1,
                                    [(x1)/2, 0., 0.],
                                    angular_stiffness = kt1,
                                    angular_damping = ct1)

joint12 = MBSLinkLinearSpringDamper(Mass1,
                                    [(x1+x2)/2, 0., 0.],
                                    Mass2,
                                    [(x1+x2)/2, 0., 0.],
                                    angular_stiffness = kt2,
                                    angular_damping = ct2)

joint23 = MBSLinkLinearSpringDamper(Mass2,
                                    [(x3+x2)/2, 0., 0.],
                                    Mass3,
                                    [(x3+x2)/2, 0., 0.],
                                    angular_stiffness = kt3,
                                    angular_damping = ct3)

mecha_sys.AddLinkage(joint01)
mecha_sys.AddLinkage(joint12)
mecha_sys.AddLinkage(joint23)

t_mbs, results = mecha_sys.RunDynamicSimulation(t_span, dt, solver_type = "constraint_stabilized")

theta1_mbs = results["Masse 1"].angles[0]
theta2_mbs = results["Masse 2"].angles[0]
theta3_mbs = results["Masse 3"].angles[0]

## Affichage des solutions

plt.figure(figsize=(11, 6))

plt.subplot(311)
plt.plot(t_eval, theta0_sig(t_eval) * 180/np.pi, label="Excitation", color = "dimgray")
plt.plot(t_eval, theta1_scipy * 180/np.pi, label="Sol. scipy")
plt.plot(t_mbs, theta1_mbs * 180/np.pi, label="Sol. MBS", ls="--")
plt.xlabel("Temps [s]")
plt.ylabel(r"Angle $\theta_1$ [°]")
plt.title(r"Barre torsion : $\theta_1$")
plt.legend()
plt.grid()

plt.subplot(312)
plt.plot(t_eval, theta2_scipy * 180/np.pi, label="Sol. scipy")
plt.plot(t_mbs, theta2_mbs * 180/np.pi, label="Sol. MBS", ls="--")
plt.xlabel("Temps [s]")
plt.ylabel(r"Angle $\theta_2$ [°]")
plt.title(r"Barre torsion : $\theta_2$")
plt.legend()
plt.grid()

plt.subplot(313)
plt.plot(t_eval, theta3_scipy * 180/np.pi, label="Sol. scipy")
plt.plot(t_mbs, theta3_mbs * 180/np.pi, label="Sol. MBS", ls="--")
plt.xlabel("Temps [s]")
plt.ylabel(r"Angle $\theta_3$ [°]")
plt.title(r"Barre torsion : $\theta_3$")
plt.legend()
plt.grid()

plt.tight_layout()
if __name__ == "__main__" :
    plt.savefig("arbre_torsion.png")


plt.figure()
plt.plot(t_mbs, theta1_mbs * 180/np.pi, label=r"$\Theta_1$")
plt.plot(t_mbs, theta2_mbs * 180/np.pi, label=r"$\Theta_2$")
plt.plot(t_mbs, theta3_mbs * 180/np.pi, label=r"$\Theta_3$")
plt.xlabel("Temps [s]")
plt.ylabel(r"Angle $\theta$ [°]")
plt.title(r"Barre en torsion")
plt.legend()
plt.grid()
plt.tight_layout()

plt.show()




