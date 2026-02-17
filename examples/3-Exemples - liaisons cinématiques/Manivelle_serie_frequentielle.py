
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from Manivelles_serie_param import *


# Définition de la plage de fréquences
freq_min = 0.3  # Hz
freq_max = 150.0  # Hz
n_freq = 1000
frequencies = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freq)
omega = 2 * np.pi * frequencies  # Pulsations [rad/s]

n_dof = 2
H_base_theo = np.zeros((n_dof, n_freq), dtype=complex)


Kb = np.array([[kx1*L1_b1**2,0] ,[0,kx1*L1_b1**2]])
Cb = np.array([[cx1*L1_b1**2,0] ,[0,cx1*L1_b1**2]])
print("Calcul en cours...")
for i, w in enumerate(omega):
    # Matrice de raideur dynamique
    Z = K_theo - w ** 2 * M_theo + 1j * w * C_theo

    # Second membre (excitation)
    F_exc = Kb + 1j * w * Cb

    # H(ω) = Z^(-1) * F_exc
    Htheo = np.linalg.solve(Z, F_exc)
    H_base_theo[:, i] = Htheo[:,0]


print("Calcul théorique terminé !")

# Extraction des FRF
H_theta1_ref1_theo = H_base_theo[0, :]  # θ1 / θ_ref1
H_theta2_ref1_theo = H_base_theo[1, :]  # θ2 / θ_ref1

# Calcul module, phase, PSD
def compute_frf_characteristics(H):
    amplitude = np.abs(H)

    phase = np.angle(H)
    phase = np.unwrap(phase)
    phase = phase - phase[0]
    phase = np.rad2deg(phase)

    psd = amplitude**2

    return amplitude, phase, psd

amp_theta1, phase_theta1, psd_theta1  = compute_frf_characteristics(H_theta1_ref1_theo)
amp_theta2, phase_theta2, psd_theta2 = compute_frf_characteristics(H_theta2_ref1_theo)




print("\nCalcul des réponses fréquentielles MBS...")
print("Configuration : Excitation sur Reference 1 (rotation Rz)")

# Fonctions de transfert à calculer :
# - θ1 / θ_ref1 : réponse de Manivelle_1 à une rotation de Reference 1
# - θ2 / θ_ref1 : réponse de Manivelle_2 à une rotation de Reference 1

spec = [
    ("Reference 1", 5, "Manivelle_1", 5),  # θ1 / θ_ref1
    ("Reference 1", 5, "Manivelle_2", 5),  # θ2 / θ_ref1
]
freqres = mecha_sys.ComputeFrequencyDomainResponse(spec,
                                                   fstart=freq_min,
                                                   fend=freq_max,)

print("Calcul MBS terminé !")


G_theta1_ref1 = freqres.SelectTransferFunctionObject_byName('Manivelle_1::θz / Reference 1::θz')
G_theta2_ref1 = freqres.SelectTransferFunctionObject_byName('Manivelle_2::θz / Reference 1::θz')


mecha_sys.ComputeQrDecomposedSystem(print_infos=True,
                                    protected_dof_spec=[("Reference 1", 0),
                                                        ("Reference 1", 1),
                                                        ("Reference 1", 5),
                                                        ("Reference 2", 0),
                                                        ("Reference 2", 1),
                                                        ("Reference 2", 5),
                                                        ("Ground", 0),
                                                        ("Ground", 1),
                                                        ("Ground", 5),])




print("\nCalcul des réponses fréquentielles MBS avec QR...")


print("Calcul MBS QR terminé !")
freqres_QR = mecha_sys.ComputeFrequencyDomainResponse(spec,
                                                   fstart=freq_min,
                                                   fend=freq_max,)

G_theta1_ref1_QR = freqres_QR.SelectTransferFunctionObject_byName('Manivelle_1::θz / Reference 1::θz')
G_theta2_ref1_QR = freqres_QR.SelectTransferFunctionObject_byName('Manivelle_2::θz / Reference 1::θz')


fig, axes = plt.subplots(3, 2, figsize=(9, 7))



# ============================================================
# Colonne 1 : θ1 / θ_ref1
# ============================================================

# Amplitude
axes[0, 0].loglog(frequencies, amp_theta1, 'b-', label='Théorie', linewidth=2)
axes[0, 0].loglog(G_theta1_ref1.frequency, G_theta1_ref1.module, 'r--',
                  label='MBS', linewidth=1.5, alpha=0.8)
axes[0, 0].loglog(G_theta1_ref1_QR.frequency, G_theta1_ref1_QR.module, 'green',
                  label='MBS QR', linewidth=1.5, alpha=0.8)

axes[0, 0].set_ylabel(r'$|H_{θ_1/θ_{ref1}}(\omega)|$')
axes[0, 0].set_title(r'FRF : $θ_1 / θ_{ref1}$ - Amplitude')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Phase
axes[1, 0].semilogx(frequencies, phase_theta1, 'b-', linewidth=2, label='Théorie')
axes[1, 0].semilogx(G_theta1_ref1.frequency, G_theta1_ref1.phase, 'r--',
                    linewidth=1.5, alpha=0.8, label='MBS')
axes[1, 0].semilogx(G_theta1_ref1_QR.frequency, G_theta1_ref1_QR.phase, 'green',
                    linewidth=1.5, alpha=0.8, label='MBS QR')

axes[1, 0].set_ylabel(r'Phase [°]')
axes[1, 0].set_title(r'FRF : $θ_1 / θ_{ref1}$ - Phase')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# PSD
axes[2, 0].loglog(frequencies, psd_theta1, 'b-', linewidth=2, label='Théorie')
axes[2, 0].loglog(G_theta1_ref1.frequency, G_theta1_ref1.powerSpectralDensity, 'r--',
                  linewidth=1.5, alpha=0.8, label='MBS')
axes[2, 0].loglog(G_theta1_ref1_QR.frequency, G_theta1_ref1_QR.powerSpectralDensity, 'green',
                  linewidth=1.5, alpha=0.8, label='MBS QR')

axes[2, 0].set_xlabel('Fréquence [Hz]')
axes[2, 0].set_ylabel(r'PSD')
axes[2, 0].set_title(r'FRF : $θ_1 / θ_{ref1}$ - PSD')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)


# ============================================================
# Colonne 2 : θ2 / θ_ref1
# ============================================================

# Amplitude
axes[0, 1].loglog(frequencies, amp_theta2, 'b-', label='Théorie', linewidth=2)
axes[0, 1].loglog(G_theta2_ref1.frequency, G_theta2_ref1.module, 'r--',
                  label='MBS', linewidth=1.5, alpha=0.8)
axes[0, 1].loglog(G_theta2_ref1_QR.frequency, G_theta2_ref1_QR.module, 'green',
                  label='MBS QR', linewidth=1.5, alpha=0.8)

axes[0, 1].set_ylabel(r'$|H_{θ_2/θ_{ref1}}(\omega)|$')
axes[0, 1].set_title(r'FRF : $θ_2 / θ_{ref1}$ - Amplitude')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Phase
axes[1, 1].semilogx(frequencies, phase_theta2, 'b-', linewidth=2, label='Théorie')
axes[1, 1].semilogx(G_theta2_ref1.frequency, G_theta2_ref1.phase, 'r--',
                    linewidth=1.5, alpha=0.8, label='MBS')
axes[1, 1].semilogx(G_theta2_ref1_QR.frequency, G_theta2_ref1_QR.phase, 'green',
                    linewidth=1.5, alpha=0.8, label='MBS QR')

axes[1, 1].set_ylabel(r'Phase [°]')
axes[1, 1].set_title(r'FRF : $θ_2 / θ_{ref1}$ - Phase')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# PSD
axes[2, 1].loglog(frequencies, psd_theta2, 'b-', linewidth=2, label='Théorie')
axes[2, 1].loglog(G_theta2_ref1.frequency, G_theta2_ref1.powerSpectralDensity, 'r--',
                  linewidth=1.5, alpha=0.8, label='MBS')
axes[2, 1].loglog(G_theta2_ref1_QR.frequency, G_theta2_ref1_QR.powerSpectralDensity, 'green',
                  linewidth=1.5, alpha=0.8, label='MBS QR')

axes[2, 1].set_xlabel('Fréquence [Hz]')
axes[2, 1].set_ylabel(r'PSD')
axes[2, 1].set_title(r'FRF : $θ_2 / θ_{ref1}$ - PSD')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('crank_system_base_excitation_frf.png', dpi=150)
# print("\nGraphique sauvegardé : 'crank_system_base_excitation_frf - penalisation 1e-5.png'")
plt.show()
