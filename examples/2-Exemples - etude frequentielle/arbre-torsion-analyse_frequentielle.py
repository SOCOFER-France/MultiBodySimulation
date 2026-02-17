import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

J1 = 1.0
J2 = 4.0
J3 = 0.5
kt1 = 10.0
kt2 = 10.0
kt3 = 4.0
ct1 = kt1 * 20/100
ct2 = kt2 * 5/100
ct3 = kt3 * 5/100

# Positions
x1 = 0.01
x2 = x1 + 0.01
x3 = x2 + 0.01


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

lambda_, phi_ = eigh(Kf, Mf) # Valeurs propres , vecteurs propres

# Normalisation  de phi_ pour obtenir
# phi_.T @ Mf @ phi_ = Identité
# phi_.T @ Kf @ phi_ = lambda_
modal_mass = np.diag(phi_.T @ Mf @ phi_)
phi_ = phi_ @ np.diag(1/np.sqrt(modal_mass))

# Calcul des fréquences propres
natural_freq_theorique = np.sqrt(lambda_) / (2*np.pi)
print("Fréquences propres du système (théorie) : ")
for f in np.sort(natural_freq_theorique) :
    print(f">> f = {f:.5e} Hz")

## Calcul théorique de la réponse fréquentielle :

def compute_amplitude_phase(H):
    amplitude = np.abs(H)
    phase = np.angle(H)
    phase = np.unwrap(phase) * 180/np.pi
    return amplitude, phase-phase[0]


# Définition de la plage de fréquences
freq_min = 0.1  # Hz
freq_max = 1.0  # Hz
n_freq = 1000
frequencies = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freq)
omega = 2 * np.pi * frequencies  # Pulsations [rad/s]

# Fonction de transfert : H(ω) = [(-ω²M + iωC + K)]^(-1) * (iωCb + Kb)
# Cette fonction donne la réponse des DDL libres à une excitation sur le DDL fixe

n_dof_free = len(freedof)
n_dof_fixed = len(fixeddof)
n_freq = len(frequencies)

# Initialisation des fonctions de transfert
H_theorique = np.zeros((n_dof_free, n_dof_fixed, n_freq), dtype=complex)

for i, w in enumerate(omega):
    # Matrice de raideur dynamique
    Z = -w ** 2 * Mf + 1j * w * Cf + Kf

    # Second membre (excitation)
    F_exc = 1j * w * Cb + Kb

    # Résolution : H(ω) = Z^(-1) * F_exc
    H_theorique[:, :, i] = np.linalg.solve(Z, F_exc)

# Extraction des fonctions de transfert individuelles
# H_theorique[i, 0, :] correspond à la réponse du DDL libre i+1 à l'excitation du DDL 0
H1_theorique = H_theorique[0, 0, :]  # Masse 1
H2_theorique = H_theorique[1, 0, :]  # Masse 2
H3_theorique = H_theorique[2, 0, :]  # Masse 3

amp1, phase1 = compute_amplitude_phase(H1_theorique)
amp2, phase2 = compute_amplitude_phase(H2_theorique)
amp3, phase3 = compute_amplitude_phase(H3_theorique)

# Calcul de la PSD (Power Spectral Density)
# PSD = |H(ω)|²
psd1 = amp1**2
psd2 = amp2**2
psd3 = amp3**2

## Etude avec MultibodySimulation
import sys, socofer
sys.path.append(socofer.lib_socofer_path)
from MultiBodySimulation.MBSBody import MBSRigidBody3D,MBSReferenceBody3D
from MultiBodySimulation.MBSMechanicalJoint import (MBSLinkLinearSpringDamper,)
from MultiBodySimulation.MBSMechanicalSystem import MBSLinearSystem

mecha_sys = MBSLinearSystem()
mecha_sys.gravity = np.array([0.0,0.0,0.0])

RefBody = MBSReferenceBody3D("Ref")
RefBody.SetReferencePosition([0., 0., 0.])

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


mecha_sys.AddRigidBody(RefBody)
mecha_sys.AddRigidBody(Mass1)
mecha_sys.AddRigidBody(Mass2)
mecha_sys.AddRigidBody(Mass3)

joint01 = MBSLinkLinearSpringDamper(RefBody,
                                    [(x1)/2, 0., 0.],
                                    Mass1,
                                    [(x1)/2, 0., 0.],
                                    angular_stiffness = [kt1,0,0],
                                    angular_damping = [ct1,0,0])

joint12 = MBSLinkLinearSpringDamper(Mass1,
                                    [(x1+x2)/2, 0., 0.],
                                    Mass2,
                                    [(x1+x2)/2, 0., 0.],
                                    angular_stiffness = [kt2,0,0],
                                    angular_damping = [ct2,0,0])

joint23 = MBSLinkLinearSpringDamper(Mass2,
                                    [(x3+x2)/2, 0., 0.],
                                    Mass3,
                                    [(x3+x2)/2, 0., 0.],
                                    angular_stiffness = [kt3,0,0],
                                    angular_damping = [ct3,0,0])

mecha_sys.AddLinkage(joint01)
mecha_sys.AddLinkage(joint12)
mecha_sys.AddLinkage(joint23)


natural_freq = mecha_sys.ComputeNaturalFrequencies(sort_values=True,
                                                    drop_zeros=True)
print("Fréquences propres du système (MBS) : ")
for f in natural_freq :
    print(f">> f = {f:.5e} Hz")




freqres = mecha_sys.ComputeFrequencyDomainResponse([
            ("Ref", 3, "Masse 1", 3),
            ("Ref", 3, "Masse 2", 3),
            ("Ref", 3, "Masse 3", 3),
                        ])

G = freqres.SelectTransferFunctionObject_byLocId(None)

print("=" * 40)
print()
## Affichage des modes propres


plt.figure(figsize=(7,8))

plt.subplot(311)
plt.title("Réponse 'Amplitude' - Comparaison MBS vs Théorie")
# Résultats MBS
plt.loglog(G.frequency, G.module, label=G.names, linestyle='--', linewidth=2)
# Résultats théoriques
plt.loglog(frequencies, amp1, label='Masse 1 (théorie)', linestyle='-', alpha=0.7)
plt.loglog(frequencies, amp2, label='Masse 2 (théorie)', linestyle='-', alpha=0.7)
plt.loglog(frequencies, amp3, label='Masse 3 (théorie)', linestyle='-', alpha=0.7)
for w0 in freqres.GetNaturalFrequencies():
    plt.axvline(w0, color="grey", alpha=0.5)
plt.ylabel(r"$|H(\omega)|$")
plt.legend(fontsize=8)
plt.grid(True)

plt.subplot(312)
plt.title("Réponse 'Phase' - Comparaison MBS vs Théorie")
# Résultats MBS
plt.semilogx(G.frequency, G.phase, label=G.names, linestyle='--', linewidth=2)
# Résultats théoriques
plt.semilogx(frequencies, phase1, label='Masse 1 (théorie)', linestyle='-', alpha=0.7)
plt.semilogx(frequencies, phase2, label='Masse 2 (théorie)', linestyle='-', alpha=0.7)
plt.semilogx(frequencies, phase3, label='Masse 3 (théorie)', linestyle='-', alpha=0.7)
for w0 in freqres.GetNaturalFrequencies():
    plt.axvline(w0, color="grey", alpha=0.5)
plt.ylabel(r"$\phi_H(\omega)$ [°]")
plt.legend(fontsize=8)
plt.grid(True)

plt.subplot(313)
plt.title("Réponse 'Power Spectral Density' - Comparaison MBS vs Théorie")
# Résultats MBS
plt.loglog(G.frequency, G.powerSpectralDensity, label=G.names, linestyle='--', linewidth=2)
# Résultats théoriques
plt.loglog(frequencies, psd1, label='Masse 1 (théorie)', linestyle='-', alpha=0.7)
plt.loglog(frequencies, psd2, label='Masse 2 (théorie)', linestyle='-', alpha=0.7)
plt.loglog(frequencies, psd3, label='Masse 3 (théorie)', linestyle='-', alpha=0.7)
for w0 in freqres.GetNaturalFrequencies():
    plt.axvline(w0, color="grey", alpha=0.5)
plt.ylabel(r"$PSD_H(\omega)$")
plt.legend(fontsize=8)
plt.grid(True)

plt.xlabel("Freq (Hz)")
plt.tight_layout()
if __name__ == "__main__":
    plt.savefig("freq_response_comparison.png")
plt.show()
