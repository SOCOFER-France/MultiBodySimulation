import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import sys
import socofer
sys.path.append(socofer.devpy_ala_path)

from Vehicle_essieux_parametrage import prepare_study




(mecha_sys,
 excitation_roue_11,excitation_roue_12,
 excitation_roue_21, excitation_roue_22,
 boite_11) = prepare_study(1e-5)

natural_freq = mecha_sys.ComputeNaturalFrequencies(sort_values=True,
                                                    drop_zeros=True,)


H_spec = [
    ("excitation 11", 2, "Caisse", 2),
    ("excitation 12", 2, "Caisse", 2),
    ("excitation 21", 2, "Caisse", 2),
    ("excitation 22", 2, "Caisse", 2),]

fstart = 1
fend = 5000
print_progress_step = 20
nbase = 1 # npoints = nbase * fmax/fmin

freqres = mecha_sys.ComputeFrequencyDomainResponse(
    H_spec,
    fstart=fstart,
    fend=fend,
    print_progress_step=print_progress_step,
    nbase=nbase,
)


G = freqres.SelectTransferFunctionObject_byLocId(None)

mecha_sys.ComputeQrDecomposedSystem(print_infos=True,
                                    print_slaves_dof=True,)


freqres = mecha_sys.ComputeFrequencyDomainResponse(
    H_spec,
    fstart=fstart,
    fend=fend,
    print_progress_step=print_progress_step,
    nbase=nbase,
)

G_qr = freqres.SelectTransferFunctionObject_byLocId(None)

plt.figure(figsize=(7,8))

plt.title("Réponse 'Amplitude' - Comparaison avec et sans QR")
# Résultats MBS
plt.loglog(G.frequency, G.module, label=G.names, linestyle='--', linewidth=2)
plt.loglog(G_qr.frequency, G_qr.module, label=[name + " - QR" for name in G.names], linestyle='-', linewidth=2)
# for w0 in natural_freq:
#     plt.axvline(w0, color="grey", alpha=0.5)
plt.ylabel(r"$|H(\omega)|$")
plt.legend(fontsize=8)
plt.grid(True)

plt.show()