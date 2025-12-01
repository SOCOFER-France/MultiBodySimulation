import numpy as np
from MultiBodySimulation.MBS_numerics import RotationMatrix

class MBSRigidBody3D:
    def __init__(self, name : str, mass : float, inertia_tensor : float | np.ndarray):
        self._name = name
        self._mass = mass

        self._is_fixed = False

        if self._mass == 0.:
            self._inv_mass = 0.
        else:
            self._inv_mass = 1 / self._mass

        # Vérification de la forme du tenseur d'inertie : doit être (3,) ou (3,3)
        if isinstance(inertia_tensor, (float, int)):
            inertia_tensor = np.eye(3) * inertia_tensor
        inertia_tensor = np.array(inertia_tensor)
        if inertia_tensor.shape == (3,):
            self._inertia = np.diag(inertia_tensor)
        elif inertia_tensor.shape == (3, 3):
            self._inertia = inertia_tensor
        else:
            raise ValueError("'inertia_tensor' must be of shape (3,) or (3,3)")



        self._referencePosition = np.zeros(3, dtype=float)
        self._refAngles = np.zeros(3, dtype=float)
        self._initial_position = np.zeros(3, dtype=float)
        self._initial_angles = np.zeros(3, dtype=float)
        self._velocity = np.zeros(3, dtype=float)
        self._omega = np.zeros(3, dtype=float)


        if np.isclose(np.abs(np.linalg.det(self._inertia)), 0.):
            self._inv_inertia = np.zeros((3, 3))
            self._inertia = np.zeros((3, 3))
        else:
            self._inv_inertia = np.linalg.inv(self._inertia)

    @property
    def IsFixed(self):
        return self._is_fixed

    @property
    def GetName(self):
        return self._name

    def GetReferencePosition(self):
        return self._referencePosition

    def GetReferenceAngle(self):
        return self._refAngles

    def SetReferencePosition(self, position : np.ndarray):
        """
        Défini la position du corps dans le repère global
        """
        position = np.asarray(position)
        if position.shape != (3,):
            raise ValueError("Position vector must be shape (3,)")

        self._referencePosition = position.copy()
        self._initial_position = position.copy()

    def SetReferenceAngle(self, angle : np.ndarray):
        """
        Défini l'angle du corps dans le repère global
        """
        angle = np.asarray(angle)
        if angle.shape != (3,):
            raise ValueError("Angle vector must be shape (3,)")

        self._refAngles = angle.copy()
        self._initial_angles = angle.copy()

    def ChangeInitialPosition(self, position : np.ndarray):
        """
        Défini la position initiale du corps dans le repère global
        """
        position = np.asarray(position)
        if position.shape != (3,):
            raise ValueError("Position vector must be shape (3,)")

        self._initial_position = position.copy()

    def ChangeInitialAngle(self, angle : np.ndarray):
        """
        Défini l'angle initial du corps dans le repère global
        """
        angle = np.asarray(angle)
        if angle.shape != (3,):
            raise ValueError("Angle vector must be shape (3,)")

        self._initial_angles = angle.copy()



    def GetBodyLocalCoords(self, global_point):
        """
        Relocalise un point en coordonnées ref en coordonnées locales du corps
        Applique la transformation
            x_local = invRotation @ (xglobal - x_cdg(ref))
        """
        global_point = np.asarray(global_point)
        if global_point.shape != (3,):
            raise ValueError("global_point must be a 3D vector of shape (3,)")
        R = RotationMatrix(*self._refAngles)
        return np.linalg.solve(R, global_point - self._referencePosition)

    def GetBodyGlobalCoords(self, local_point):
        """
        Relocalise un point en coordonnées locales du corps en coordonnées du référentiel
        Applique la transformation
            x_ref = x_cdg(ref) + Rotation @ x_local
        """
        local_point = np.asarray(local_point)
        if local_point.shape != (3,):
            raise ValueError("local_point must be a 3D vector of shape (3,)")
        R = RotationMatrix(*self._refAngles)

        return self._referencePosition + R @ local_point

    def GetBodyGlobalVector(self, local_vector):
        """
        Transforme un vecteur dans les coordonnées locales d'un corps vers le référentiel
        Applique la transformation
            v_ref = Rotation @ v_local
        """
        local_vector = np.array(local_vector)
        if local_vector.shape != (3,):
            raise ValueError("local_vector must be a 3D vector of shape (3,)")
        R = RotationMatrix(*self._refAngles)
        return R @ local_vector

class MBSReferenceBody3D(MBSRigidBody3D) :

    def __init__(self,name):

        super().__init__(name,mass=0.,
                              inertia_tensor=0.)
        self._is_fixed = True

        self.__zero_func = lambda t : np.zeros_like(t, dtype=float)

        self.__dX_func = self.__zero_func
        self.__dY_func = self.__zero_func
        self.__dZ_func = self.__zero_func

        self.__dthetaX_func = self.__zero_func
        self.__dthetaY_func = self.__zero_func
        self.__dthetaZ_func = self.__zero_func

        self.__VX_func = self.__zero_func
        self.__VY_func = self.__zero_func
        self.__VZ_func = self.__zero_func

        self._omegaX_func = self.__zero_func
        self._omegaY_func = self.__zero_func
        self._omegaZ_func = self.__zero_func


    def SetDisplacementFunction(self,
                                 dx_func=None,
                                 dy_func=None,
                                 dz_func=None):
        dh = 1e-9
        if dx_func is not None :
            self.__dX_func = dx_func
            self.__VX_func = lambda t : (dx_func(t+dh) - dx_func(t-dh) ) / (2*dh)

        if dy_func is not None :
            self.__dY_func = dy_func
            self.__VY_func = lambda t : (dy_func(t+dh) - dy_func(t-dh) ) / (2*dh)

        if dz_func is not None :
            self.__dZ_func = dz_func
            self.__VZ_func = lambda t : (dz_func(t+dh) - dz_func(t-dh) ) / (2*dh)


    def SetRotationFunction(self,dtheta_x_func=None,
                             dtheta_y_func=None,
                             dtheta_z_func=None):
        dh = 1e-9
        if dtheta_x_func is not None :
            self.__dthetaX_func = dtheta_x_func
            self._omegaX_func = lambda t : (dtheta_x_func(t+dh) - dtheta_x_func(t-dh) ) / (2*dh)

        if dtheta_y_func is not None :
            self.__dthetaY_func = dtheta_y_func
            self.__domegaY_func = lambda t : (dtheta_y_func(t+dh) - dtheta_y_func(t-dh) ) / (2*dh)

        if dtheta_z_func is not None :
            self.__dthetaZ_func = dtheta_z_func
            self._omegaZ_func = lambda t : (dtheta_z_func(t+dh) - dtheta_z_func(t-dh) ) / (2*dh)


    def _updateDisplacement(self,t):

        return np.array([
            self.__dX_func(t),
            self.__dY_func(t),
            self.__dZ_func(t),
            self.__dthetaX_func(t),
            self.__dthetaY_func(t),
            self.__dthetaZ_func(t),
            self.__VX_func(t),
            self.__VY_func(t),
            self.__VZ_func(t),
            self._omegaX_func(t),
            self._omegaY_func(t),
            self._omegaZ_func(t)
        ])






