
import numpy as np


import sys
import socofer
sys.path.append(socofer.devpy_ala_path)

from MultiBodySimulation.MBSBody import MBSRigidBody3D,MBSReferenceBody3D
from MultiBodySimulation.MBSMechanicalJoint import MBSLinkLinearSpringDamper, MBSLinkKinematic
from MultiBodySimulation.MBSMechanicalSystem import MBSLinearSystem


kinematic_tolerance = 1e-5


Lx_loco = 16.0 # Longueur totale
Hz_loco = 1.0
Hy_loco = 2.0
Lx_loco_essieu = 10.0 # Longueur entre essieux

L_essieu = 1.5 # Longueur essieu
L_boite = 1000e-3 # longueur de boite
H_boite = 200e-3 # hauteur et largeur de boite
Ly_roue = 1435e-3

K_spring = 1200e3 # N/m
C_spring = K_spring * 0.01 #1%
K_lat_spring = 1500
C_lat_spring = K_lat_spring * 0.01

rho = 7800 # densité acier

De = 150e-3 # Diametre essieu
Dr = 920e-3 # Diametre roue
Lr = 150e-3

Me = rho * L_essieu * np.pi * De**2/4
Mr = rho * Lr * np.pi * Dr**2/4

Jy_ess = 1/2 * Me * De**2/4
Jy_roue = 1/2 * Mr * Dr**2/4

Jx_ess = 1/4 * Me * De**2/4 + 1/12 * Me * L_essieu**2
Jx_roue = 1/4 * Mr * Dr**2/4 + 1/12 * Mr * Lr**2

Mb = L_boite * H_boite**2 * rho
Jxb = Mb/12 * (2*H_boite**2)
Jyb = Mb/12 * (L_boite**2 + H_boite**2)
Jzb = Mb/12 * (L_boite**2 + H_boite**2)

Munsuspended = Mb * 4 + Me * 2 + Mr * 4
Mloc = 30e3 - Munsuspended
Jx_loc = Mloc/12 * (Hy_loco**2 + Hz_loco**2)
Jy_loc = Mloc/12 * (Lx_loco**2 + Hz_loco**2)
Jz_loc = Mloc/12 * (Hy_loco**2 + Lx_loco**2)

print(f"Masse non suspendue : { Munsuspended:.2f} kg")
print(f"Masse caisse : {Mloc :.2f} kg")


# Solides

caisse_loco = MBSRigidBody3D("Caisse", Mloc, [Jx_loc, Jy_loc, Jz_loc])
caisse_loco.SetReferencePosition([0., 0., 2.2])

essieu_1 = MBSRigidBody3D("Essieu 1", Me + 2*Mr,
                          [Jx_ess+2*Jx_roue,
                           Jy_ess+2*Jy_roue,
                           Jx_ess+2*Jx_roue]) # Jx = Jz
essieu_2 = MBSRigidBody3D("Essieu 2", Me + 2*Mr,
                          [Jx_ess+2*Jx_roue,
                           Jy_ess+2*Jy_roue,
                           Jx_ess+2*Jx_roue]) # Jx = Jz
essieu_1.SetReferencePosition([-Lx_loco_essieu/2, 0., Dr/2])
essieu_2.SetReferencePosition([ Lx_loco_essieu/2, 0., Dr/2])

boite_11 = MBSRigidBody3D("Boite 11", Mb, [Jxb, Jyb, Jzb])
boite_12 = MBSRigidBody3D("Boite 12", Mb, [Jxb, Jyb, Jzb])
boite_21 = MBSRigidBody3D("Boite 21", Mb, [Jxb, Jyb, Jzb])
boite_22 = MBSRigidBody3D("Boite 22", Mb, [Jxb, Jyb, Jzb])

boite_11.SetReferencePosition([-Lx_loco_essieu/2, -L_essieu/2, Dr/2])
boite_12.SetReferencePosition([-Lx_loco_essieu/2,  L_essieu/2, Dr/2])
boite_21.SetReferencePosition([ Lx_loco_essieu/2, -L_essieu/2, Dr/2])
boite_22.SetReferencePosition([ Lx_loco_essieu/2,  L_essieu/2, Dr/2])


excitation_roue_11 = MBSReferenceBody3D("excitation 11")
excitation_roue_12 = MBSReferenceBody3D("excitation 12")
excitation_roue_21 = MBSReferenceBody3D("excitation 21")
excitation_roue_22 = MBSReferenceBody3D("excitation 22")

excitation_roue_11.SetReferencePosition([-Lx_loco_essieu/2, -Ly_roue/2, Dr/2])
excitation_roue_12.SetReferencePosition([-Lx_loco_essieu/2,  Ly_roue/2, Dr/2])
excitation_roue_21.SetReferencePosition([ Lx_loco_essieu/2, -Ly_roue/2, Dr/2])
excitation_roue_22.SetReferencePosition([ Lx_loco_essieu/2,  Ly_roue/2, Dr/2])



# Liaisons
pivot_boite_11 = MBSLinkKinematic(boite_11,
                               boite_11.GetReferencePosition(),
                               essieu_1,
                               boite_11.GetReferencePosition(),
                                Tx = 1,Rx = 1,
                                Ty = 1,Ry = 0,
                                Tz = 1,Rz = 1,
                                  kinematic_tolerance = kinematic_tolerance)
pivot_boite_12 = MBSLinkKinematic(boite_12,
                               boite_12.GetReferencePosition(),
                               essieu_1,
                               boite_12.GetReferencePosition(),
                                Tx = 1,Rx = 1,
                                Ty = 1,Ry = 0,
                                Tz = 1,Rz = 1,
                                  kinematic_tolerance = kinematic_tolerance)
pivot_boite_21 = MBSLinkKinematic(boite_21,
                               boite_21.GetReferencePosition(),
                               essieu_2,
                               boite_21.GetReferencePosition(),
                                Tx = 1,Rx = 1,
                                Ty = 1,Ry = 0,
                                Tz = 1,Rz = 1,
                                  kinematic_tolerance = kinematic_tolerance)
pivot_boite_22 = MBSLinkKinematic(boite_22,
                               boite_22.GetReferencePosition(),
                               essieu_2,
                               boite_22.GetReferencePosition(),
                                Tx = 1,Rx = 1,
                                Ty = 1,Ry = 0,
                                Tz = 1,Rz = 1,
                                  kinematic_tolerance = kinematic_tolerance)


fixation_roue_11 = MBSLinkKinematic(excitation_roue_11,
                               excitation_roue_11.GetReferencePosition(),
                               essieu_1,
                               excitation_roue_11.GetReferencePosition(),
                                Tz = 1,
                                  kinematic_tolerance = kinematic_tolerance)
fixation_roue_12 = MBSLinkKinematic(excitation_roue_12,
                               excitation_roue_12.GetReferencePosition(),
                               essieu_1,
                               excitation_roue_12.GetReferencePosition(),
                                Tz = 1,
                                  kinematic_tolerance = kinematic_tolerance)
fixation_roue_21 = MBSLinkKinematic(excitation_roue_21,
                               excitation_roue_21.GetReferencePosition(),
                               essieu_2,
                               excitation_roue_21.GetReferencePosition(),
                                Tz = 1,
                                  kinematic_tolerance = kinematic_tolerance)
fixation_roue_22 = MBSLinkKinematic(excitation_roue_22,
                               excitation_roue_22.GetReferencePosition(),
                               essieu_2,
                               excitation_roue_22.GetReferencePosition(),
                                Tz = 1,
                                  kinematic_tolerance = kinematic_tolerance)



# Ressorts
Kmat = [K_lat_spring, K_lat_spring, K_spring]
Cmat = [C_lat_spring, C_lat_spring, C_spring]
delta_x_boite =  np.array([L_boite/2,0,0])

ressort_1_boite_11 = MBSLinkLinearSpringDamper(boite_11,
                       boite_11.GetReferencePosition() + delta_x_boite,
                               caisse_loco,
                       boite_11.GetReferencePosition() + delta_x_boite,
                        stiffness = Kmat,
                        damping = Cmat)
ressort_2_boite_11 = MBSLinkLinearSpringDamper(boite_11,
                       boite_11.GetReferencePosition() - delta_x_boite,
                               caisse_loco,
                       boite_11.GetReferencePosition() - delta_x_boite,
                        stiffness = Kmat,
                        damping = Cmat)

ressort_1_boite_12 = MBSLinkLinearSpringDamper(boite_12,
                       boite_12.GetReferencePosition() + delta_x_boite,
                               caisse_loco,
                       boite_12.GetReferencePosition() + delta_x_boite,
                        stiffness = Kmat,
                        damping = Cmat)
ressort_2_boite_12 = MBSLinkLinearSpringDamper(boite_12,
                       boite_12.GetReferencePosition() - delta_x_boite,
                               caisse_loco,
                       boite_12.GetReferencePosition() - delta_x_boite,
                        stiffness = Kmat,
                        damping = Cmat)

ressort_1_boite_21 = MBSLinkLinearSpringDamper(boite_21,
                       boite_21.GetReferencePosition() + delta_x_boite,
                               caisse_loco,
                       boite_21.GetReferencePosition() + delta_x_boite,
                        stiffness = Kmat,
                        damping = Cmat)
ressort_2_boite_21 = MBSLinkLinearSpringDamper(boite_21,
                       boite_21.GetReferencePosition() - delta_x_boite,
                               caisse_loco,
                       boite_21.GetReferencePosition() - delta_x_boite,
                        stiffness = Kmat,
                        damping = Cmat)

ressort_1_boite_22 = MBSLinkLinearSpringDamper(boite_22,
                       boite_22.GetReferencePosition() + delta_x_boite,
                               caisse_loco,
                       boite_22.GetReferencePosition() + delta_x_boite,
                        stiffness = Kmat,
                        damping = Cmat)
ressort_2_boite_22 = MBSLinkLinearSpringDamper(boite_22,
                       boite_22.GetReferencePosition() - delta_x_boite,
                               caisse_loco,
                       boite_22.GetReferencePosition() - delta_x_boite,
                        stiffness = Kmat,
                        damping = Cmat)


# Assemblage
mecha_sys = MBSLinearSystem()
mecha_sys.AddRigidBody(caisse_loco)
mecha_sys.AddRigidBody( essieu_1 )
mecha_sys.AddRigidBody( essieu_2 )
mecha_sys.AddRigidBody( boite_11 )
mecha_sys.AddRigidBody( boite_12 )
mecha_sys.AddRigidBody( boite_21 )
mecha_sys.AddRigidBody( boite_22 )
mecha_sys.AddRigidBody( excitation_roue_11 )
mecha_sys.AddRigidBody( excitation_roue_12 )
mecha_sys.AddRigidBody( excitation_roue_21 )
mecha_sys.AddRigidBody( excitation_roue_22 )

mecha_sys.AddLinkage( pivot_boite_11 )
mecha_sys.AddLinkage( pivot_boite_12 )
mecha_sys.AddLinkage( pivot_boite_21 )
mecha_sys.AddLinkage( pivot_boite_22 )
mecha_sys.AddLinkage( fixation_roue_11 )
mecha_sys.AddLinkage( fixation_roue_12 )
mecha_sys.AddLinkage( fixation_roue_21 )
mecha_sys.AddLinkage( fixation_roue_22 )

mecha_sys.AddLinkage( ressort_1_boite_11 )
mecha_sys.AddLinkage( ressort_2_boite_11 )
mecha_sys.AddLinkage( ressort_1_boite_12 )
mecha_sys.AddLinkage( ressort_2_boite_12 )
mecha_sys.AddLinkage( ressort_1_boite_21 )
mecha_sys.AddLinkage( ressort_2_boite_21 )
mecha_sys.AddLinkage( ressort_1_boite_22 )
mecha_sys.AddLinkage( ressort_2_boite_22 )

mecha_sys.CheckUnconstrainedDegreeOfFreedom()