import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import sys, socofer
sys.path.append(socofer.devpy_ala_path)

from MultiBodySimulation.MBSBody import MBSRigidBody3D, MBSReferenceBody3D
from MultiBodySimulation.MBSMechanicalJoint import (MBSLinkLinearSpringDamper,
                                                     MBSLinkKinematic)
from MultiBodySimulation.MBSMechanicalSystem import MBSLinearSystem

# =============================================================================
# PARAMÈTRES PHYSIQUES
# =============================================================================
m = 1.0           # masse de la bielle [kg]
L = 0.5           # longueur de la bielle [m]
J = m * L**2 / 12 # inertie au CDG (barre uniforme) [kg·m²]
Jeq = J + (L/2)**2 * m

k_ressort = 50.0  # raideur du ressort [N/m]
c_ressort = 0.5   # amortissement du ressort [N·s/m]

f = 0.5 # Hz
theta_amp = 5 * np.pi / 180
h_amp = L * np.sin(theta_amp)

h_signal = lambda t: np.sin(2 * np.pi * f * t) * h_amp + np.sin(2 * np.pi * 1.5*f * t + np.pi/5) * h_amp / 3
vh_signal = lambda t : 2 * np.pi * f * h_amp * np.cos(2 * np.pi * f * t) + \
                    2 * np.pi * 3*f * np.cos(2 * np.pi * 1.5*f * t + np.pi/5) * h_amp / 3



t_span = [0, 10.0]
dt = 0.01
t_eval = np.linspace(t_span[0], t_span[1], 1000)

print(f"\n{'='*70}")
print("BIELLE AVEC PIVOT ET RESSORT - Validation des couplages")
print(f"{'='*70}")
print(f"\nParamètres physiques:")
print(f"  Masse bielle     : m = {m} kg")
print(f"  Longueur         : L = {L} m")
print(f"  Inertie au CDG   : J = {J:.6f} kg·m²")
print(f"  Raideur ressort  : k = {k_ressort} N/m")
print(f"  Amortissement    : c = {c_ressort} N·s/m")

# =============================================================================
# SOLUTION ANALYTIQUE (PETITS ANGLES)
# =============================================================================
print(f"\n{'='*70}")
print("SOLUTION ANALYTIQUE")
print(f"{'='*70}")

# Équation du mouvement (petits angles) :
# Quand la bielle tourne de θ, l'extrémité se déplace en Y de : y = L·sin(θ) ≈ L·θ
# Force du ressort : F_y = -k·y = -k·L·θ
# Moment au pivot : M_pivot = L·F_y = -k·L²·θ
# Moment au CDG : M_CDG = (L/2)·F_y = -k·L²/2·θ
#
# Équation : Jeq·θ̈ + c·(L²/2)·θ̇ + k·(L²/2)·θ = 0
# Jeq == J + (L/2)^2 * M

# =============================================================================
# SOLUTION SCIPY (VÉRIFICATION)
# =============================================================================
print(f"\n{'='*70}")
print("SOLUTION SCIPY (vérification équation)")
print(f"{'='*70}")

def bielle_ode(t, y):
    theta, omega = y
    # Équation : Jeq·θ̈ = -k·(L²/2)·θ - c·(L²/2)·ω
    h0 = h_signal(t)
    v0 = vh_signal(t)

    h1 = L * np.sin(theta)
    v1 = L * omega

    Text = L * (k_ressort * (h0 - h1) + c_ressort * (v0 - v1))
    alpha = Text / (Jeq)
    return [omega, alpha]

sol_scipy = solve_ivp(
    bielle_ode,
    t_span,
    [0.0, 0.0],
    t_eval=t_eval,
    method='RK45'
)

theta_scipy_ode = sol_scipy.y[0]
y_ext_scipy_ode = L * np.sin(theta_scipy_ode)
x_ext_scipy_ode = L * np.cos(theta_scipy_ode)

y_ext_scipy_smallAngles = L * theta_scipy_ode
x_ext_scipy_smallAngles = L/2 * (1 + np.cos(theta_scipy_ode))


# =============================================================================
# MODÉLISATION MBS
# =============================================================================
print(f"\n{'='*70}")
print("MODÉLISATION MULTIBODY SIMULATION")
print(f"{'='*70}")

# Système MBS
sys_mbs = MBSLinearSystem()
sys_mbs.gravity = np.array([0., 0., 0.])  # Pas de gravité

# Corps fixe (pivot)
pivot_body = MBSReferenceBody3D("Pivot")
pivot_body.SetReferencePosition([0., 0., 0.])

# Corps fixe (reference)
reference_body = MBSReferenceBody3D("Reference")
reference_body.SetReferencePosition([L, 0., 0.])
reference_body.SetDisplacementFunction(dy_func = h_signal)


# Corps mobile (bielle)
bielle = MBSRigidBody3D("Bielle", mass=m, inertia_tensor=J)
bielle.SetReferencePosition([L/2, 0., 0.])  # CDG au milieu

# Liaison pivot (bloque translations, bloque rotations X et Y, laisse Z libre)
liaison_pivot = MBSLinkKinematic(
    pivot_body, [0., 0., 0.],
    bielle, [0., 0., 0.],
    Tx=1, Ty=1, Tz=1,  # Translations bloquées
    Rx=1, Ry=1,        # Rotations X, Y bloquées
    Rz=0,              # Rotation Z libre (autour de Z)
)

# Ressort à l'extrémité (position de référence [L, 0, 0])
# En petits angles, le déplacement Y de l'extrémité ≈ L·θ
ressort = MBSLinkLinearSpringDamper(
    reference_body, [L, 0., 0.],
    bielle, [L, 0., 0.],
    stiffness=[0., k_ressort, 0.],  # Ressort en Y seulement
    damping=[0., c_ressort, 0.]
)

# Assemblage
sys_mbs.AddRigidBody(pivot_body)
sys_mbs.AddRigidBody(reference_body)
sys_mbs.AddRigidBody(bielle)
sys_mbs.AddLinkage(liaison_pivot)
sys_mbs.AddLinkage(ressort)

sys_mbs.AssemblyMatrixSystem(print_report=True)


# =============================================================================
# SIMULATION
# =============================================================================
print(f"\n{'='*70}")
print("SIMULATION")
print(f"{'='*70}")

import time
t_start = time.time()
t_mbs, results = sys_mbs.RunDynamicSimulation(
    t_span=t_span,
    dt=dt,
    solver_type = "constraint_stabilized")

t_end = time.time()

print(f"\nTemps de calcul : {t_end - t_start:.3f} s")

# Extraction des résultats
bielle_results = results["Bielle"]
theta_mbs = bielle_results.angles[2]
# Position de l'extrémité (reconstruction)
ext_mbs = bielle_results.get_connected_point_motion([L,0,0], approx_rotation=False)
y_ext_mbs = ext_mbs.positions[1]
x_ext_mbs = ext_mbs.positions[0]

# =============================================================================
# GRAPHIQUES
# =============================================================================
fig, axes = plt.subplots(3,1, figsize=(7, 7))

# Graphique 1 : Angle θ(t)
axes[0].plot(t_eval, np.rad2deg(theta_scipy_ode), 'g-',
                label='Scipy ODE - largeAngles', linewidth=2.0)
axes[0].plot(t_mbs, np.rad2deg(theta_mbs), 'r-',
                label='MBS', linewidth=1)
axes[0].set_xlabel('Temps [s]')
axes[0].set_ylabel('Angle θ [°]')
axes[0].set_title('Rotation de la bielle')
axes[0].legend()
axes[0].grid(True)

# Graphique 2 : Position Y de l'extrémité
axes[1].plot(t_eval, y_ext_scipy_ode * 1000, 'g-',
                label='Scipy ODE - largeAngles', linewidth=2.0)
axes[1].plot(t_eval, y_ext_scipy_smallAngles * 1000, 'b-',
                label='Scipy ODE - smallAngles', linewidth=2.0)
axes[1].plot(t_mbs, y_ext_mbs * 1000, 'r-',
                label='MBS', linewidth=1)
axes[1].set_xlabel('Temps [s]')
axes[1].set_ylabel('Position Y extrémité [mm]')
axes[1].set_title('Déplacement de l\'extrémité (couplage θ → Y)')
axes[1].legend()
axes[1].grid(True)

# Graphique 2 : Position X de l'extrémité
axes[2].plot(t_eval, x_ext_scipy_ode * 1000, 'g-',
                label='Scipy ODE - largeAngles', linewidth=2.0)
axes[2].plot(t_eval, x_ext_scipy_smallAngles * 1000, 'b-',
                label='Scipy ODE - smallAngles', linewidth=2.0)
axes[2].plot(t_mbs, x_ext_mbs * 1000, 'r-',
                label='MBS', linewidth=1)
axes[2].set_xlabel('Temps [s]')
axes[2].set_ylabel('Position X extrémité [mm]')
axes[2].set_title('Déplacement de l\'extrémité (couplage θ → X)')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
if __name__ == "__main__" :
    plt.savefig("bielle_reaction.png")
    plt.show()
