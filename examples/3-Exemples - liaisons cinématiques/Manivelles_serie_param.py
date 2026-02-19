import numpy as np
import sys, socofer
sys.path.append(socofer.devpy_ala_path)

from MultiBodySimulation.MBSBody import MBSRigidBody3D, MBSReferenceBody3D
from MultiBodySimulation.MBSMechanicalJoint import (
    MBSLinkKinematic,
    MBSLinkLinearSpringDamper,
)
from MultiBodySimulation.MBSMechanicalSystem import MBSLinearSystem



# ============================================================
# PARAMÈTRES PHYSIQUES ET GÉOMÉTRIQUES
# ============================================================
kinematic_tolerance = 1e-6
# Géométrie des manivelles
L1_b1 = 0.10  # Longueur bras 1 de la manivelle 1 [m]
L2_b1 = 0.15  # Longueur bras 2 de la manivelle 1 [m]
L1_b2 = 0.15  # Longueur bras 1 de la manivelle 2 [m]
L2_b2 = 0.18  # Longueur bras 2 de la manivelle 2 [m]

# Distance entre les centres des deux manivelles
d12 = 0.30  # [m]

# Propriétés d'inertie
Jz1 = 0.10  # Moment d'inertie manivelle 1 [kg.m²]
Jz2 = 0.14  # Moment d'inertie manivelle 2 [kg.m²]
mass1 = 2.0  # Masse manivelle 1 [kg]
mass2 = mass1*Jz2/Jz1  # Masse manivelle 2 [kg]

# Raideurs des ressorts
kx1 = 100.0  # Raideur ressort manivelle 1 - référence [N/m]
k12 = 50.0   # Raideur ressort entre manivelles [N/m]
kx2 = 80.0   # Raideur ressort manivelle 2 - référence [N/m]

# Amortissements (5% de la raideur pour avoir un amortissement raisonnable)
cx1 = kx1 * 0.05  # [N.s/m]
c12 = k12 * 0.05  # [N.s/m]
cx2 = kx2 * 0.05  # [N.s/m]


# ============================================================
# CRÉATION DU SYSTÈME MÉCANIQUE
# ============================================================

mecha_sys = MBSLinearSystem()
mecha_sys.gravity = np.array([0.0, 0.0, 0.0])  # Pas de gravité pour simplifier


# ============================================================
# CRÉATION DES CORPS
# ============================================================

# Corps de référence (sol)
ground = MBSReferenceBody3D("Ground")
mecha_sys.AddRigidBody(ground)

RefBody1 = MBSReferenceBody3D("Reference 1")
RefBody1.SetReferencePosition([0.0, 0.0, 0.0])
mecha_sys.AddRigidBody(RefBody1)

RefBody2 = MBSReferenceBody3D("Reference 2")
RefBody2.SetReferencePosition([d12, 0.0, 0.0])
mecha_sys.AddRigidBody(RefBody2)

# Manivelle 1 - positionnée à l'origine
Crank1 = MBSRigidBody3D(
    "Manivelle_1",
    mass=mass1,
    inertia_tensor=np.diag([Jz1/2, Jz1/2, Jz1])  # Jx, Jy, Jz
)
Crank1.SetReferencePosition([0.0, 0.0, 0.0])
mecha_sys.AddRigidBody(Crank1)

# Manivelle 2 - positionnée à distance d12 selon x
Crank2 = MBSRigidBody3D(
    "Manivelle_2",
    mass=mass2,
    inertia_tensor=np.diag([Jz2/2, Jz2/2, Jz2])  # Jx, Jy, Jz
)
Crank2.SetReferencePosition([d12, 0.0, 0.0])
mecha_sys.AddRigidBody(Crank2)


# ============================================================
# LIAISONS CINÉMATIQUES (PIVOTS D'AXE Z)
# ============================================================

# Pivot pour la manivelle 1 (rotation libre autour de z)
pivot1 = MBSLinkKinematic(
    ground,
    Crank1.GetReferencePosition(),  # Au cdg de la manivelle 1
    Crank1,
    Crank1.GetReferencePosition(),  # Au cdg de la manivelle 1
    Tx=1, Rx=1,  # Bloqué en translation x et rotation x
    Ty=1, Ry=1,  # Bloqué en translation y et rotation y
    Tz=1, Rz=0,   # Bloqué en translation z, libre en rotation z (pivot)
    kinematic_tolerance=kinematic_tolerance,
)
mecha_sys.AddLinkage(pivot1)

# Pivot pour la manivelle 2 (rotation libre autour de z)
pivot2 = MBSLinkKinematic(
    ground,
    Crank2.GetReferencePosition(),  # Au cdg de la manivelle 2
    Crank2,
    Crank2.GetReferencePosition(),  # Au cdg de la manivelle 2
    Tx=1, Rx=1,  # Bloqué en translation x et rotation x
    Ty=1, Ry=1,  # Bloqué en translation y et rotation y
    Tz=1, Rz=0,   # Bloqué en translation z, libre en rotation z (pivot)
    kinematic_tolerance=kinematic_tolerance,
)
mecha_sys.AddLinkage(pivot2)


# ============================================================
# RESSORTS DE COUPLAGE
# ============================================================

# Ressort 1 : Bras 1 de manivelle 1 vers référence
# Point sur manivelle 1 en coordonnées locales : (0, L1_b1, 0)
# Point sur référence (position d'équilibre du ressort) : (0, L1_b1, 0)
spring1 = MBSLinkLinearSpringDamper(
    Crank1,
    [0.0, L1_b1, 0.0],  # Point d'attache sur manivelle 1
    RefBody1,
    [0.0, L1_b1, 0.0],  # Point d'attache sur référence
    stiffness=[kx1, 0.0, 0.0],  # Ressort actif en x uniquement
    damping=[cx1, 0.0, 0.0]
)
mecha_sys.AddLinkage(spring1)

# Ressort 2 : Bras 2 de manivelle 1 vers bras 1 de manivelle 2
# Point sur manivelle 1 : (0, -L2_b1, 0)
# Point sur manivelle 2 : (0, -L1_b2, 0)
# Position d'équilibre : entre les deux manivelles
spring12 = MBSLinkLinearSpringDamper(
    Crank1,
    [d12, -L2_b1, 0.0],  # Point d'attache sur manivelle 1
    Crank2,
    [d12, -L1_b2, 0.0],  # Point d'attache sur manivelle 2
    stiffness=[k12, 0.0, 0.0],  # Ressort actif en x uniquement
    damping=[c12, 0.0, 0.0]
)
mecha_sys.AddLinkage(spring12)

# Ressort 3 : Bras 2 de manivelle 2 vers référence
# Point sur manivelle 2 : (0, L2_b2, 0)
# Point sur référence : (d12, L2_b2, 0)
spring2 = MBSLinkLinearSpringDamper(
    Crank2,
    [0.0, L2_b2, 0.0],  # Point d'attache sur manivelle 2
    RefBody2,
    [d12, L2_b2, 0.0],  # Point d'attache sur référence
    stiffness=[kx2, 0.0, 0.0],  # Ressort actif en x uniquement
    damping=[cx2, 0.0, 0.0]
)
mecha_sys.AddLinkage(spring2)


# ============================================================
# VÉRIFICATION DU SYSTÈME
# ============================================================

print("="*60)
print("SYSTÈME À DEUX MANIVELLES COUPLÉES - PARAMÉTRAGE")
print("="*60)
print(f"\nNombre de corps rigides : {len(mecha_sys.bodies)}")
print(f"\nNombre de corps référence : {len(mecha_sys.ref_bodies)}")
print(f"Nombre de liaisons : {len(mecha_sys.links)}")
# Calcul et affichage des fréquences propres
print("\n--- Analyse modale ---")

natural_freq = mecha_sys.ComputeNaturalFrequencies(sort_values=True, drop_zeros=True )

print("Fréquences propres du système :")
for i, f in enumerate(natural_freq, 1):
    print(f"  Mode {i} : f = {f:.5f} Hz")

print("\n" + "="*60)
print("Système paramétré avec succès !")
print("="*60)



# ============================================================
# CONSTRUCTION DES MATRICES THÉORIQUES
# ============================================================

print("="*60)
print("CONSTRUCTION DES MATRICES DU SYSTÈME")
print("="*60)

# Matrice de masse (2x2)
M_theo = np.array([[Jz1, 0.0],
                   [0.0, Jz2]])

# Matrice de raideur (2x2)
# K11 = kx1*L1_b1² + k12*L2_b1²
# K12 = K21 = -k12*L2_b1*L1_b2
# K22 = k12*L1_b2² + kx2*L2_b2²
K_theo = np.array([[kx1*L1_b1**2 + k12*L2_b1**2, -k12*L2_b1*L1_b2],
                   [-k12*L2_b1*L1_b2, k12*L1_b2**2 + kx2*L2_b2**2]])

# Matrice d'amortissement (2x2)
C_theo = np.array([[cx1*L1_b1**2 + c12*L2_b1**2, -c12*L2_b1*L1_b2],
                   [-c12*L2_b1*L1_b2, c12*L1_b2**2 + cx2*L2_b2**2]])

print("\nMatrice de masse M :")
print(M_theo)
print("\nMatrice de raideur K :")
print(K_theo)
print("\nMatrice d'amortissement C :")
print(C_theo)


# ============================================================
# ANALYSE MODALE THÉORIQUE
# ============================================================

from scipy.linalg import eigh

lambda_theo, phi_theo = eigh(K_theo, M_theo)
natural_freq_theo = np.sqrt(lambda_theo) / (2*np.pi)

print("\n" + "="*60)
print("ANALYSE MODALE THÉORIQUE")
print("="*60)
print("Fréquences propres (théorie) :")
for i, f in enumerate(natural_freq_theo, 1):
    print(f"  Mode {i} : f = {f:.5f} Hz")
