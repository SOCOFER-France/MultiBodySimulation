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

t_span = [0, 30]
t_eval = np.linspace(t_span[0], t_span[1], 1000)
dt = t_eval[1] - t_eval[0]

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

t_mbs, results = mecha_sys.RunDynamicSimulation(t_span, dt)

natural_freq = mecha_sys.ComputeNaturalFrequencies(sort_values=True,
                                                    drop_zeros=True)
print("Fréquences propres du système (MBS) : ")
for f in natural_freq :
    print(f">> f = {f:.5e} Hz")



mbs_modal_result = mecha_sys.ComputeModalAnalysis(sort_values = True,
                                                            drop_zeros = True,)

modal_result_dict = mbs_modal_result.GetDisplacementsByBodies()
freq_vector_mbs = mbs_modal_result.GetNaturalFrequencies()


freqres = mecha_sys.ComputeFrequencyDomainResponse([
            ("Ref", 3, "Masse 1", 3),
            ("Ref", 3, "Masse 2", 3),
            ("Ref", 3, "Masse 3", 3)
                        ])

G = freqres.SelectTransferFunctionObject_byLocId(None)

print("=" * 40)
print()
## Affichage des modes propres

fig = plt.figure()
ax = fig.add_subplot(111)

color = []
for i, mode in enumerate( phi_ ) :
    l = ax.plot(natural_freq_theorique, np.rad2deg(mode), label=f"Theorique masse {i}",
                    marker = "s",alpha = 0.5, ls="", markeredgecolor="k",ms=10)[0]
    color.append( l.get_color() )

k = 0
for masse, mode in modal_result_dict.items() :
    ax.plot(freq_vector_mbs, np.rad2deg(mode[3]), label=f"MBS : {masse}",
                    marker = "o", ls = "", color=color[k])
    k+=1

ax.grid()
ax.set(title="Déplacements des modes propres",
       xlabel = "Fréquence [Hz]",
       ylabel = "Angle [°]")

plt.figure(figsize=(7,8))
plt.subplot(311)
plt.loglog(G.frequency, G.module, label = G.names)
for w0 in freqres.GetNaturalFrequencies() :
    plt.axvline(w0, color = "grey")
plt.legend()
plt.grid(True)

plt.subplot(312)
plt.semilogx(G.frequency, G.phase, label = G.names)
for w0 in freqres.GetNaturalFrequencies() :
    plt.axvline(w0, color = "grey")
plt.legend()
plt.grid(True)

plt.subplot(313)
plt.loglog(G.frequency, G.powerSpectralDensity, label = G.names)
for w0 in freqres.GetNaturalFrequencies() :
    plt.axvline(w0, color = "grey")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

