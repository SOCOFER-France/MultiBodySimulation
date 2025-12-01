from msilib.schema import Property

import numpy as np


from MultiBodySimulation.MBSBody import MBSRigidBody3D

class _MBSLink3D:
    def __init__(self,
                    body1  : MBSRigidBody3D,
                    global_point1 : np.ndarray,
                    body2  : MBSRigidBody3D,
                    global_point2 : np.ndarray,
                    stiffness : (np.ndarray, float) = None,
                    damping : (np.ndarray, float) = None,
                    angular_stiffness : (np.ndarray, float) = None,
                    angular_damping : (np.ndarray, float) = None,
                    linear_reaction = True,
                    has_freegap = False,
                    kinematic_link = False):
        if stiffness is None and damping is None and  angular_stiffness is None and angular_damping is None:
            raise ValueError("All stiffness and damping values (translation or rotations) can't be None")

        self._linear_reaction = linear_reaction
        self._freegap = has_freegap
        self._kinematic_link = kinematic_link

        self.__body1 = body1
        self.__body2 = body2


        global_point1 = np.asarray(global_point1)
        global_point2 = np.asarray(global_point2)
        if global_point1.shape != (3,) or global_point2.shape != (3,):
            raise ValueError("Attachment points must be 3D vectors")


        self.__local_point1 = body1.GetBodyLocalCoords(global_point1)
        self.__local_point2 = body2.GetBodyLocalCoords(global_point2)



        self._K = np.zeros((3,3))
        self._C = np.zeros((3, 3))
        self._K_theta = np.zeros((3, 3))
        self._C_theta = np.zeros((3, 3))

        self._init_structural_stiffness(stiffness, damping, angular_stiffness, angular_damping)

        self._TransGap_vector = None
        self._RotGap_vector = None

    @property
    def GetBody1(self):
        return self.__body1

    @property
    def GetBody2(self):
        return self.__body2

    @property
    def GetGlobalPoint1(self):
        return self.__body1.GetBodyGlobalCoords(self.__local_point1)

    @property
    def GetGlobalPoint2(self):
        return self.__body2.GetBodyGlobalCoords(self.__local_point2)


    @property
    def HasGap(self):
        return self._freegap

    @property
    def GetTransGap(self):
        return self._TransGap_vector

    @property
    def GetRotGap(self):
        return self._RotGap_vector


    @property
    def IsLinear(self):
        return self._linear_reaction


    @property
    def IsKinematic(self):
        return self._kinematic_link

    @property
    def GetLinearReactionMatrices(self):
        return self._K, self._C, self._K_theta, self._C_theta

    def _init_structural_stiffness(self, stiffness, damping, angular_stiffness, angular_damping):

        if stiffness is None:
            stiffness = 0.0

        if isinstance(stiffness, (float, int)):
            self._K = np.eye(3) * float(stiffness)
        else:
            self._K = np.array(stiffness)
            if self._K.shape == (3,):
                self._K = np.eye(3) * self._K
            elif self._K.shape != (3, 3):
                raise ValueError("Stiffness must be scalar, 3 element array or 3x3 array")

        if damping is None:
            damping = 0.0

        if isinstance(damping, (float, int)):
            self._C = np.eye(3) * float(damping)
        else:
            self._C = np.array(damping)
            if self._C.shape == (3,):
                self._C = np.eye(3) * self._C
            elif self._C.shape != (3, 3):
                raise ValueError("Damping must be scalar, 3 element array or 3x3 array")

        if angular_stiffness is None:
            angular_stiffness = 0.

        if isinstance(angular_stiffness, (float, int)):
            self._K_theta = np.eye(3) * float(angular_stiffness)
        else:
            self._K_theta = np.asarray(angular_stiffness)
            if self._K_theta.shape == (3,):
                self._K_theta = np.eye(3) * self._K_theta
            elif self._K_theta.shape != (3, 3):
                raise ValueError("Angular stiffness must be scalar, 3 element array or 3x3 array")

        if angular_damping is None:
            angular_damping = 0.

        if isinstance(angular_damping, (float, int)):
            self._C_theta = np.eye(3) * float(angular_damping)
        else:
            self._C_theta = np.array(angular_damping)
            if self._C_theta.shape == (3,):
                self._C_theta = np.eye(3) * self._C_theta
            elif self._C_theta.shape != (3, 3):
                raise ValueError("Angular damping must be scalar, 3 element array or 3x3 array")


    def _check_gap_values(self, Tx, name=""):
        error_mess = f"Incorrect hard stop value in direction ({name}) : needs float/int or 2-values tuple of float / int"
        if Tx is None :
            return 0.0, (-np.inf, np.inf)
        elif isinstance(Tx, (float,int)):
            return 1.0, (-Tx, Tx)
        elif isinstance(Tx, (tuple, list)):
            if len(Tx) == 2 and all(isinstance(tx, (float, int)) for tx in Tx) :
                return 1.0, (min(Tx), max(Tx))
            else :
                raise ValueError(error_mess)
        else :
            raise ValueError(error_mess)

    def GetLinearLocalReactions(self, U1, V1, U2, V2):
        dp = U2[:3] - U1[:3]
        dv = V2[:3] - V1[:3]
        angle_axis = U2[3:] - U1[3:]
        omega_rel = V2[3:] - V1[3:]

        # Force de ressort + amortisseur
        if self._K is not None or self._C is not None:
            force = self._K @ dp + self._C @ dv  # Vers body2
        else :
            force = np.zeros(3)

        # Couple des liaisons angulaires
        if self._K_theta is not None or self._C_theta is not None:
            # Calcul du moment (torque) de ressort et d'amortissement angulaire
            torque_rot = self._K_theta @ angle_axis + self._C_theta @ omega_rel
        else :
            torque_rot = np.zeros(3)

        return force, torque_rot

    def GetNonLinearLocalReactions(self, U1=None, V1=None, U2=None, V2=None,
                                        dUlocal=None, dVlocal=None):
        return np.zeros(3), np.zeros(3)


class MBSLinkLinearSpringDamper(_MBSLink3D) :

    def __init__(self,
        body1: MBSRigidBody3D,
        global_point1: np.ndarray,
        body2: MBSRigidBody3D,
        global_point2: np.ndarray,
        stiffness: (np.ndarray, float) = None,
        damping: (np.ndarray, float) = None,
        angular_stiffness: (np.ndarray, float) = None,
        angular_damping: (np.ndarray, float) = None,):


        super().__init__(body1,
                        global_point1,
                        body2,
                        global_point2,
                        stiffness,
                        damping,
                        angular_stiffness,
                        angular_damping,
                       linear_reaction=True,
                       has_freegap=False,
                       kinematic_link=False
                       )

class MBSLinkKinematic(_MBSLink3D):

    def __init__(self,
                        body1: MBSRigidBody3D, global_point1: np.ndarray,
                        body2: MBSRigidBody3D, global_point2: np.ndarray,
                        Tx : int=0.,
                        Ty : int=0.,
                        Tz : int=0.,
                        Rx : int=0.,
                        Ry : int=0.,
                        Rz : int=0.,
                        kinematic_tolerance=1e-4,
                 ):
        self.__kinematicConstraints = np.array([Tx, Ty, Tz, Rx, Ry, Rz])

        if ((self.__kinematicConstraints != 0) & (self.__kinematicConstraints != 1)).any() :
            raise ValueError("Kinematic contraints (Tx, Ty, Tz, Rx, Ry, Rz) must be 0 or +1")

        M = max(body1._mass, body2._mass)
        Jm = max(np.max(body1._inertia), np.max(body2._inertia))
        k = M / kinematic_tolerance
        c = 2 * np.sqrt(k * M)
        ktheta = Jm / (2 * np.pi * kinematic_tolerance)
        ctheta = 2 * np.sqrt(ktheta * Jm)

        Kmat = np.diag([Tx, Ty, Tz]) * k
        Cmat = np.diag([Tx, Ty, Tz]) * c
        Ktheta_mat = np.diag([Rx, Ry, Rz]) * ktheta
        Ctheta_mat = np.diag([Rx, Ry, Rz]) * ctheta

        super().__init__(body1,
                         global_point1,
                         body2,
                         global_point2,
                         Kmat,
                         Cmat,
                         Ktheta_mat,
                         Ctheta_mat,
                         linear_reaction=True,
                         has_freegap=False,
                         kinematic_link=True
                         )

        self.__translationFriction = None
        self.__rotationFriction = None

    @property
    def GetTransConstraints(self):
        return self.__kinematicConstraints[:3]

    @property
    def GetRotConstraints(self):
        return self.__kinematicConstraints[:3]

    def SetTransFriction(self, normalInitialContactForces: float = 0.,
                         frictionCoefficient: float = 0.2,
                         relativeSpeedRegulationScale: float = 1e-3,
                         normalProjectionMatrix : np.ndarray = None):
        """
        Ajoute un frottement aux translations

        :param:

        - normalInitialContactForces : force normale initiale (corps 2 >>> corps 1) noté Fn_init;
        - frictionCoefficient : coefficient de frottement noté µ;
        - relativeSpeedRegulationScale : vitesse pour lissage de la loi de Coulomb noté vs;
        - normalProjectionMatrix : matrice de projection pour les efforts normaux notés P >> Fn = P @ F;

        Pour un frottement sur une direction normale n, utilisez P = n . n^T
        Si P la matrice P est None :
         - P = diag(Tx,Ty,Tz) ==> translations bloquées

        Définition de la loi  :
        Les efforts sont définis pour corps 2 >>> corps 1

        La matrice de projection tangentielle Q = id - P
        Efforts de réactions locales de contact normaux
        Fn_reac = norm( P @ ( K dX + X dV ) )

        Efforts normaux
        Fn = Fn_reac + Fn_init

        vitesse tangentielle
        v = Q @ dv

        Efforts locaux de frottement tangents
        Ft = µ * Fn * tanh(norm(v) / vs) * direction

        avec ct un damping pour hautes fréquences.

        """
        if normalProjectionMatrix is None :
            normalProjectionMatrix = np.diag(self.__kinematicConstraints[:3])

        tangentialProjectMatrix = np.eye(3) - normalProjectionMatrix

        self.__translationFriction = [normalInitialContactForces,
                                      frictionCoefficient,
                                      relativeSpeedRegulationScale,
                                      normalProjectionMatrix,
                                      tangentialProjectMatrix]
        self.__linear_reaction = False

    def SetRotationalFriction(self, rotationRadius : float,
                              normalInitialContactForces: float = 0.,
                              frictionCoefficient: float = 0.2,
                              relativeOmegaRegulationScale: float = 1e-3,
                              normalProjectionMatrix: np.ndarray = None):
        """
        Ajoute un frottement aux translations

        :param:

        - rotationRadius : rayon interne de la liaison
        - normalInitialContactForces : force normale initiale (corps 2 >>> corps 1) noté Fn_init;
        - frictionCoefficient : coefficient de frottement noté µ;
        - relativeOmegaRegulationScale : vitesse angulaire pour lissage de la loi de Coulomb noté ws;
        - normalProjectionMatrix : matrice de projection pour des efforts notés P;

        Pour un frottement sur une direction normale n, utilisez P = n . n^T
        Si la matrice de projection est None :
            - P = diag(Tx,Ty,Tz) ==> translations bloquées
        La matrice de projection tangentielle est Q = Qrot = Ptrans = P

        Définition de la loi  :
        Les efforts sont définis pour corps 2 >>> corps 1
        Efforts de réactions locales de contact normaux
        Fn_reac = norm( P @ ( K dX + C dV ) )

        Efforts normaux
        Fn = Fn_reac + Fn_init

        Vitesse tangentielle
        w = P @ omega = Q @ omega


        Couple de frottement
        Tf = µ * r * Fn * tanh(norm(w) / ws) * direction(w)

        avec ct un damping pour hautes fréquences.

        """

        if normalProjectionMatrix is None :
            normalProjectionMatrix = np.diag(self.__kinematicConstraints[:3])


        self.__rotationFriction = [rotationRadius,
                                   normalInitialContactForces,
                                   frictionCoefficient,
                                   relativeOmegaRegulationScale,
                                   normalProjectionMatrix]
        self.__linear_reaction = False

    def GetNonLinearLocalReactions(self, U1=None, V1=None, U2=None, V2=None,
                                        dUlocal=None, dVlocal=None):
        local_disp = True
        if U1 is None or U2 is None or V1 is None or V1 is None :
            local_disp = False
        local_deflection = True
        if dUlocal is None or dVlocal is None :
            local_deflection = False

        if (local_disp and local_deflection) or not(local_disp or local_deflection) :
            raise ValueError("Parameter error")

        if self.__rotationFriction is None and self.__translationFriction is None :
            # Dans les faits ce cas n'arrivera pas
            return np.zeros(3), np.zeros(3)

        if local_disp :
            dp = U2[:3] - U1[:3]
            dv = V2[:3] - V1[:3]
            omega_rel = V2[3:] - V1[3:]
        elif local_deflection :
            dp = dUlocal[:3]
            dv = dVlocal[:3]
            omega_rel = dVlocal[:3]
        else :
            raise ValueError("Parameter error")

        # Force de ressort + amortisseur
        if self.K is not None or self.C is not None:
            force = self._K @ dp + self._C @ dv  # Vers body2
        else:
            force = np.zeros(3)

        Ft = np.zeros(3)
        if self.__translationFriction is not None :
            (Fn_init,
             µ,
             vs,
             P,
             Q) = self.__translationFriction

            vt = Q @ dv
            v_norm = np.linalg.norm(vt)
            if v_norm > 1e-12:
                direction = vt / v_norm
            else:
                direction = np.zeros_like(vt)

            ct = 1e-3 * Fn_init / vs


            Fn_reac = np.linalg.norm( P @ force )
            Fn_tot = (Fn_init + Fn_reac)

            ct = 1e-3 * Fn_tot * µ / vs
            fvs = np.tanh(v_norm / vs)
            Ft = - Fn_tot * µ * fvs * direction + vt * ct

        Tf = np.zeros(3)
        if self.__rotationFriction is not None :
            (r,
             Fn_init,
             µ,
             ws,
             P) = self.__rotationFriction

            wt = P @ omega_rel
            w_norm = np.linalg.norm(wt)
            if w_norm > 1e-12:
                direction = wt / w_norm
            else:
                direction = np.zeros_like(wt)

            Fn_reac = np.linalg.norm( P @ force )
            Fn_tot = (Fn_init + Fn_reac)

            ct = 1e-3 * Fn_tot * r * µ / ws
            fws = np.tanh(w_norm/ws)
            Tf = - Fn_tot * r * µ * fws * direction + wt * ct

        return Ft, Tf





class MBSLinkHardStop(_MBSLink3D) :
    def __init__(self, body1: MBSRigidBody3D, global_point1: np.ndarray,
                 body2: MBSRigidBody3D, global_point2: np.ndarray,
                 Tx_gap=None,
                 Ty_gap=None,
                 Tz_gap=None,
                 Rx_gap=None,
                 Ry_gap=None,
                 Rz_gap=None,
                 penetration_tolerance=1e-4,
                 ):

        super().__init__(body1,
                         global_point1,
                         body2,
                         global_point2,
                         0,
                         0,
                         0,
                         0,
                         linear_reaction=True,
                         has_freegap=True,
                         kinematic_link=False
                         )



        M = max(body1._mass, body2._mass)
        Jm = max(np.max(body1._inertia), np.max(body2._inertia))
        k = M / penetration_tolerance
        c = 2 * np.sqrt(k * M)
        ktheta = Jm / (2 * np.pi * penetration_tolerance)
        ctheta = 2 * np.sqrt(ktheta * Jm)


        Tx, Tx_gap = self._check_gap_values(Tx_gap, "X")
        Ty, Ty_gap = self._check_gap_values(Ty_gap, "Y")
        Tz, Tz_gap = self._check_gap_values(Tz_gap, "Z")
        Rx, Rx_gap = self._check_gap_values(Rx_gap, "thetaX")
        Ry, Ry_gap = self._check_gap_values(Ry_gap, "thetaY")
        Rz, Rz_gap = self._check_gap_values(Rz_gap, "thetaZ")

        Kmat = np.diag([Tx, Ty, Tz]) * k
        Cmat = np.diag([Tx, Ty, Tz]) * c
        Ktheta_mat = np.diag([Rx, Ry, Rz]) * ktheta
        Ctheta_mat = np.diag([Rx, Ry, Rz]) * ctheta


        self._init_structural_stiffness(Kmat,Cmat,Ktheta_mat, Ctheta_mat)


        self._TransGap_vector = np.array([Tx_gap, Ty_gap, Tz_gap])
        self._RotGap_vector = np.array([Rx_gap, Ry_gap, Rz_gap])

    def GetLinearLocalReactions(self, U1, V1, U2, V2):
        dp = U2[:3] - U1[:3]
        dv = V2[:3] - V1[:3]
        angle_axis = U2[3:] - U1[3:]
        omega_rel = V2[3:] - V1[3:]

        du_minus = self._TransGap_vector[:, 0]
        du_plus = self._TransGap_vector[:, 1]
        du = np.maximum(0., dp - du_plus) + np.minimum(0., dp - du_minus)
        dv = dv * (np.abs(du) > 0.)

        dtheta_minus = self._RotGap_vector[:,0]
        dtheta_plus = self._TransGap_vector[:, 1]
        dtheta = np.maximum(0., angle_axis - dtheta_plus) + np.minimum(0., angle_axis - dtheta_minus)
        domega = omega_rel * (np.abs(dtheta)>0.0)

        # Force de ressort + amortisseur
        if self._K is not None or self._C is not None:
            force = self._K @ du + self._C @ dv  # Vers body2
        else:
            force = np.zeros(3)

        # Couple des liaisons angulaires
        if self._K_theta is not None or self._C_theta is not None:
            # Calcul du moment (torque) de ressort et d'amortissement angulaire
            torque_rot = self._K_theta @ dtheta + self._C_theta @ domega
        else:
            torque_rot = np.zeros(3)

        return force, torque_rot



class MBSLinkSmoothLinearStop(_MBSLink3D) :
    def __init__(self, body1: MBSRigidBody3D, global_point1: np.ndarray,
                 body2: MBSRigidBody3D, global_point2: np.ndarray,
                 stiffness,
                 damping,
                 angular_stiffness,
                 angular_damping,
                 Tx_gap=None,
                 Ty_gap=None,
                 Tz_gap=None,
                 Rx_gap=None,
                 Ry_gap=None,
                 Rz_gap=None,
                 ):
        super().__init__(body1,
                         global_point1,
                         body2,
                         global_point2,
                         stiffness,
                         damping,
                         angular_stiffness,
                         angular_damping,
                         linear_reaction=True,
                         has_freegap=True,
                         kinematic_link=False
                         )


        Tx, Tx_gap = self._check_gap_values(Tx_gap, "X")
        Ty, Ty_gap = self._check_gap_values(Ty_gap, "Y")
        Tz, Tz_gap = self._check_gap_values(Tz_gap, "Z")
        Rx, Rx_gap = self._check_gap_values(Rx_gap, "thetaX")
        Ry, Ry_gap = self._check_gap_values(Ry_gap, "thetaY")
        Rz, Rz_gap = self._check_gap_values(Rz_gap, "thetaZ")




        self._TransGap_vector = np.array([Tx_gap, Ty_gap, Tz_gap])
        self._RotGap_vector = np.array([Rx_gap, Ry_gap, Rz_gap])


        self._K = np.diag([Tx, Ty, Tz]) * self._K
        self._C = np.diag([Tx, Ty, Tz]) * self._C
        self._K_theta = np.diag([Rx, Ry, Rz]) * self._K_theta
        self._C_theta = np.diag([Rx, Ry, Rz]) * self._C_theta

    def GetLinearLocalReactions(self, U1, V1, U2, V2):
        dp = U2[:3] - U1[:3]
        dv = V2[:3] - V1[:3]
        angle_axis = U2[3:] - U1[3:]
        omega_rel = V2[3:] - V1[3:]

        du_minus = self._TransGap_vector[:, 0]
        du_plus = self._TransGap_vector[:, 1]
        du = np.maximum(0., dp - du_plus) + np.minimum(0., dp - du_minus)
        dv = dv * (np.abs(du) > 0.)

        dtheta_minus = self._RotGap_vector[:,0]
        dtheta_plus = self._TransGap_vector[:, 1]
        dtheta = np.maximum(0., angle_axis - dtheta_plus) + np.minimum(0., angle_axis - dtheta_minus)
        domega = omega_rel * (np.abs(dtheta)>0.0)

        # Force de ressort + amortisseur
        if self._K is not None or self._C is not None:
            force = self._K @ du + self._C @ dv  # Vers body2
        else:
            force = np.zeros(3)

        # Couple des liaisons angulaires
        if self._K_theta is not None or self._C_theta is not None:
            # Calcul du moment (torque) de ressort et d'amortissement angulaire
            torque_rot = self._K_theta @ dtheta + self._C_theta @ domega
        else:
            torque_rot = np.zeros(3)

        return force, torque_rot