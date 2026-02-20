"""MBSBody
--------

Module contenant les classes représentant des corps rigides 3D utilisés dans
la simulation multi-corps.

Classes principales:
- :class:`MBSRigidBody3D` : corps rigide 3D avec masse et inertie.
- :class:`MBSReferenceBody3D` : corps de référence fixe avec cinématique imposée.

"""

import numpy as np
from MultiBodySimulation.MBS_numerics import RotationMatrix

class MBSRigidBody3D:
    """Corps rigide 3D avec masse et inertie pour simulation multi-corps.

    Cette classe représente un corps rigide 3D caractérisé par sa masse,
    son tenseur d'inertie, sa position de référence, ses angles de rotation,
    et ses vitesses associées. Elle fournit des méthodes pour convertir entre
    coordonnées globales et coordonnées locales du corps.

    Paramètres du constructeur (:class:`__init__`):

    :param name: nom identifiant le corps
    :type name: str
    :param mass: masse du corps (kg)
    :type mass: float
    :param inertia_tensor: tenseur d'inertie (scalaire, vecteur 3 ou matrice 3x3)
    :type inertia_tensor: float | numpy.ndarray

    Attributs principaux:

    :var _mass: masse du corps
    :var _inv_mass: inverse de la masse (0 si masse nulle)
    :var _inertia: tenseur d'inertie (matrice 3x3)
    :var _inv_inertia: inverse du tenseur d'inertie
    :var _referencePosition: position du centre de masse en coordonnées globales
    :var _refAngles: angles de rotation (Euler XYZ) en coordonnées globales
    :var _initial_position: position initiale du corps
    :var _initial_angles: angles initiaux du corps
    :var _velocity: vitesse du centre de masse
    :var _omega: vitesse angulaire
    :var _is_fixed: indique si le corps est fixe (immobile)
    """
    
    def __init__(self, name : str, mass : float, inertia_tensor : float | np.ndarray):
        self._name = name
        if name == "Linkage" :
            raise ValueError("Name 'Linkage' is not allowed for bodies")
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
        """Indique si le corps est fixe (immobile).

        :return: True si le corps est fixe, False sinon
        :rtype: bool
        """
        return self._is_fixed

    @property
    def GetName(self):
        """Retourne le nom du corps.

        :return: nom du corps
        :rtype: str
        """
        return self._name

    def GetReferencePosition(self):
        """Retourne la position de référence du centre de masse du corps.

        :return: vecteur position (x, y, z) en coordonnées globales
        :rtype: numpy.ndarray
        """
        return self._referencePosition

    def GetReferenceAngle(self):
        """Retourne les angles de rotation du corps.

        :return: angles (Euler XYZ) en radians
        :rtype: numpy.ndarray
        """
        return self._refAngles

    def SetReferencePosition(self, position : np.ndarray):
        """Défini la position du centre de masse du corps dans le repère global.

        Met à jour à la fois la position de référence et la position initiale
        du corps.

        :param position: vecteur position (x, y, z) en coordonnées globales
        :type position: numpy.ndarray
        :return: None

        :raises ValueError: si le vecteur position n'est pas de forme (3,)
        """
        position = np.asarray(position)
        if position.shape != (3,):
            raise ValueError("Position vector must be shape (3,)")

        self._referencePosition = position.copy()
        self._initial_position = position.copy()

    def SetReferenceAngle(self, angle : np.ndarray):
        """Défini les angles de rotation du corps dans le repère global.

        Met à jour à la fois les angles de référence et les angles initiaux
        du corps. Utilise la convention Euler XYZ (en radians).

        :param angle: vecteur d'angles (theta_x, theta_y, theta_z) en radians
        :type angle: numpy.ndarray
        :return: None

        :raises ValueError: si le vecteur d'angle n'est pas de forme (3,)
        """
        angle = np.asarray(angle)
        if angle.shape != (3,):
            raise ValueError("Angle vector must be shape (3,)")

        self._refAngles = angle.copy()
        self._initial_angles = angle.copy()

    def ChangeInitialPosition(self, position : np.ndarray):
        """Modifie la position initiale du corps.

        :param position: vecteur position initiale (x, y, z) en coordonnées globales
        :type position: numpy.ndarray
        :return: None

        :raises ValueError: si le vecteur position n'est pas de forme (3,)
        """
        position = np.asarray(position)
        if position.shape != (3,):
            raise ValueError("Position vector must be shape (3,)")

        self._initial_position = position.copy()

    def ChangeInitialAngle(self, angle : np.ndarray):
        """Modifie les angles initiaux du corps (convention Euler XYZ).

        :param angle: vecteur d'angles initiaux (theta_x, theta_y, theta_z) en radians
        :type angle: numpy.ndarray
        :return: None

        :raises ValueError: si le vecteur d'angle n'est pas de forme (3,)
        """
        angle = np.asarray(angle)
        if angle.shape != (3,):
            raise ValueError("Angle vector must be shape (3,)")

        self._initial_angles = angle.copy()



    def GetBodyLocalCoords(self, global_point):
        """Convertit un point des coordonnées globales aux coordonnées locales du corps.

        Applique la transformation inverse de rotation :

        .. math::

            \\mathbf{x}_{local} = R^{-1} (\\mathbf{x}_{global} - \\mathbf{x}_{ref})

        où :math:`R` est la matrice de rotation du corps.

        :param global_point: point en coordonnées globales (vecteur 3D)
        :type global_point: numpy.ndarray
        :return: point en coordonnées locales du corps
        :rtype: numpy.ndarray

        :raises ValueError: si global_point n'est pas un vecteur 3D
        """
        global_point = np.asarray(global_point)
        if global_point.shape != (3,):
            raise ValueError("global_point must be a 3D vector of shape (3,)")
        R = RotationMatrix(*self._refAngles)
        return np.linalg.solve(R, global_point - self._referencePosition)

    def GetBodyGlobalCoords(self, local_point):
        """Convertit un point des coordonnées locales du corps aux coordonnées globales.

        Applique la transformation de rotation :

        .. math::

            \\mathbf{x}_{global} = \\mathbf{x}_{ref} + R \\mathbf{x}_{local}

        où :math:`R` est la matrice de rotation du corps et :math:`\\mathbf{x}_{ref}`
        est la position du centre de masse.

        :param local_point: point en coordonnées locales du corps (vecteur 3D)
        :type local_point: numpy.ndarray
        :return: point en coordonnées globales
        :rtype: numpy.ndarray

        :raises ValueError: si local_point n'est pas un vecteur 3D
        """
        local_point = np.asarray(local_point)
        if local_point.shape != (3,):
            raise ValueError("local_point must be a 3D vector of shape (3,)")
        R = RotationMatrix(*self._refAngles)

        return self._referencePosition + R @ local_point

    def GetBodyGlobalVector(self, local_vector):
        """Transforme un vecteur des coordonnées locales aux coordonnées globales.

        Cette méthode applique uniquement la rotation sans translation, ce qui
        est approprié pour les vecteurs (contrairement aux points).

        .. math::

            \\mathbf{v}_{global} = R \\mathbf{v}_{local}

        où :math:`R` est la matrice de rotation du corps.

        :param local_vector: vecteur en coordonnées locales du corps (vecteur 3D)
        :type local_vector: numpy.ndarray
        :return: vecteur en coordonnées globales
        :rtype: numpy.ndarray

        :raises ValueError: si local_vector n'est pas un vecteur 3D
        """
        local_vector = np.array(local_vector)
        if local_vector.shape != (3,):
            raise ValueError("local_vector must be a 3D vector of shape (3,)")
        R = RotationMatrix(*self._refAngles)
        return R @ local_vector

class MBSReferenceBody3D(MBSRigidBody3D):
    """Corps de référence fixe avec cinématique imposée.

    Un corps de référence est un corps fixe (immobile) auquel on peut imposer
    une cinématique arbitraire via des fonctions du temps. Contrairement aux
    corps rigides ordinaires, sa position et sa rotation sont entièrement
    contrôlées par les fonctions d'entrée et ne sont pas affectées par les
    forces de liaison.

    Cette classe est typiquement utilisée pour modéliser des excitations
    imposées (par exemple, un moteur, une table vibrante, etc.).

    Paramètres du constructeur (:class:`__init__`):

    :param name: nom identifiant le corps de référence
    :type name: str

    Attributs:

    :var _name: nom du corps
    :var _mass: toujours 0 (corps fixe)
    :var _inertia: toujours nulle (corps fixe)
    :var _is_fixed: toujours True

    Méthodes importantes:
    - :meth:`SetDisplacementFunction` pour imposer le déplacement (translation).
    - :meth:`SetRotationFunction` pour imposer la rotation.
    - :meth:`_updateDisplacement` pour obtenir l'état cinématique au temps t.
    """

    def __init__(self, name: str):
        """Initialise un corps de référence fixe.

        :param name: nom identifiant le corps de référence
        :type name: str
        """
        super().__init__(name, mass=0.,
                         inertia_tensor=0.)
        self._is_fixed = True

        self.__zero_func = lambda t: np.zeros_like(t, dtype=float)

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
        """Impose les fonctions de déplacement du corps de référence.

        Défini les déplacements en translation selon X, Y et/ou Z en fonction
        du temps. Les vitesses sont calculées numériquement par différence finie
        (schéma centré d'ordre 2).

        :param dx_func: fonction temps → déplacement selon X. Signature: ``dx_func(t) -> float``
        :type dx_func: callable | None
        :param dy_func: fonction temps → déplacement selon Y. Signature: ``dy_func(t) -> float``
        :type dy_func: callable | None
        :param dz_func: fonction temps → déplacement selon Z. Signature: ``dz_func(t) -> float``
        :type dz_func: callable | None
        :return: None
        """
        dh = 1e-9
        if dx_func is not None:
            self.__dX_func = dx_func
            self.__VX_func = lambda t: (dx_func(t + dh) - dx_func(t - dh)) / (2 * dh)

        if dy_func is not None:
            self.__dY_func = dy_func
            self.__VY_func = lambda t: (dy_func(t + dh) - dy_func(t - dh)) / (2 * dh)

        if dz_func is not None:
            self.__dZ_func = dz_func
            self.__VZ_func = lambda t: (dz_func(t + dh) - dz_func(t - dh)) / (2 * dh)


    def SetRotationFunction(self, dtheta_x_func=None,
                             dtheta_y_func=None,
                             dtheta_z_func=None):
        """Impose les fonctions de rotation du corps de référence.

        Défini les rotations (angles Euler XYZ) en fonction du temps.
        Les vitesses angulaires (omega) sont calculées numériquement par
        différence finie.

        :param dtheta_x_func: fonction temps → rotation autour X (radians). Signature: ``dtheta_x_func(t) -> float``
        :type dtheta_x_func: callable | None
        :param dtheta_y_func: fonction temps → rotation autour Y (radians). Signature: ``dtheta_y_func(t) -> float``
        :type dtheta_y_func: callable | None
        :param dtheta_z_func: fonction temps → rotation autour Z (radians). Signature: ``dtheta_z_func(t) -> float``
        :type dtheta_z_func: callable | None
        :return: None
        """
        dh = 1e-9
        if dtheta_x_func is not None:
            self.__dthetaX_func = dtheta_x_func
            self._omegaX_func = lambda t: (dtheta_x_func(t + dh) - dtheta_x_func(t - dh)) / (2 * dh)

        if dtheta_y_func is not None:
            self.__dthetaY_func = dtheta_y_func
            self.__domegaY_func = lambda t: (dtheta_y_func(t + dh) - dtheta_y_func(t - dh)) / (2 * dh)

        if dtheta_z_func is not None:
            self.__dthetaZ_func = dtheta_z_func
            self._omegaZ_func = lambda t: (dtheta_z_func(t + dh) - dtheta_z_func(t - dh)) / (2 * dh)


    def _updateDisplacement(self, t):
        """Retourne l'état cinématique complet du corps de référence au temps t.

        Récupère tous les déplacements, angles et vitesses imposés à partir
        des fonctions définies par :meth:`SetDisplacementFunction` et
        :meth:`SetRotationFunction`.

        :param t: temps
        :type t: float
        :return: vecteur d'état [dx, dy, dz, dtheta_x, dtheta_y, dtheta_z,
                 vx, vy, vz, omega_x, omega_y, omega_z]
        :rtype: numpy.ndarray
        """
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

    def ChangeInitialPosition(self, *args, **kwargs):
        """Surcharge pour empêcher la modification de position initiale.

        Les corps de référence ont leur cinématique imposée ; modifier la
        position initiale n'aurait aucun effet. Cette méthode affiche un
        message d'avertissement.

        :return: None
        """
        print("Changer la position initiale d'un corps de référence n'est pas possible. "
              "Cette commande n'a pas d'effet.")

    def ChangeInitialAngle(self, *args, **kwargs):
        """Surcharge pour empêcher la modification des angles initiaux.

        Les corps de référence ont leur cinématique imposée ; modifier les
        angles initiaux n'aurait aucun effet. Cette méthode affiche un
        message d'avertissement.

        :return: None
        """
        print("Changer l'angle initial d'un corps de référence n'est pas possible. "
              "Cette commande n'a pas d'effet.")






