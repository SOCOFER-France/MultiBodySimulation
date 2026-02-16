import warnings

import numpy as np
from typing import List

from MultiBodySimulation.MBSBody import MBSRigidBody3D
from MultiBodySimulation.MBS_numerics import RotationMatrix, ApproxRotationMatrix

class MBSBodySimulationResult :
    def __init__(self, body: MBSRigidBody3D,
                 time_eval : np.array,
                 displacement_array : np.array,
                 position_array : np.array,
                 velocity_array : np.array):
        """
        body : corps
        time_eval : array des temps ;
        displacement_array : déplacements (translation et angles) par rapport à la position de référence
        position_array : position dans le temps (ref_pos + disp)
        velocity_array : vitesse (translation et angles)
        """
        self.body = body
        self.name = self.body.GetName

        self.time_eval = time_eval
        self.positions = position_array[0:3]
        self.angles = position_array[3:6]
        self.displacements = displacement_array[0:3]
        self.angle_displacements = displacement_array[3:6]
        self.velocities = velocity_array[0:3]
        self.omega = velocity_array[3:6]

        self.accelerations = np.gradient(self.velocities, self.time_eval, edge_order=2, axis=1)
        self.gamma = np.gradient(self.omega, self.time_eval, edge_order=2, axis=1)


    def get_connected_point_motion(self,global_point, approx_rotation=False):
        """
        Calcul le déplacement, la vitesse et l'accélération d'un point appartenant au solide
        """
        global_point = np.asarray(global_point)
        if global_point.shape != (3,):
            raise ValueError("global_point must be a 3D vector of shape (3,)")

        local_point = self.body.GetBodyLocalCoords(global_point)

        point_x = np.zeros_like(self.positions)
        point_v = np.zeros_like(self.velocities)
        point_acc = np.zeros_like(self.accelerations)
        nt = self.positions.shape[1]

        for i in range(nt) :
            angles = self.angles[:,i]
            if approx_rotation :
                R = ApproxRotationMatrix(*angles)
            else :
                R = RotationMatrix(*angles)

            point_x[:,i] = self.positions[:,i] + R @ local_point
            point_v[:,i] = self.positions[:,i] + np.cross(self.omega[:,i], R @ local_point)
            if i > 0 :
                point_acc[:,i] = (point_v[:,i] - point_v[:,i-1]) / (self.time_eval[i] - self.time_eval[i-1])

        return MBSDistantPointMotionResults(self.time_eval,
                                            point_x,
                                            point_v,
                                            point_acc)

class MBSDistantPointMotionResults :

    def __init__(self, time_eval, positions, velocities, accelerations):
        self.time_eval = time_eval
        self.positions = positions
        self.velocities = velocities
        self.accelerations = accelerations


class MBSModalResults :

    def __init__(self, natural_pulsations : np.array,
                       modal_displacements : np.array,
                       body_names_list : List[str]):


        self.__naturalPulsations = natural_pulsations
        self.__modalDisplacements = modal_displacements
        self.__bodiesNames = body_names_list

        self.__nmodes = len(natural_pulsations)
        self.__nbodies = len(body_names_list)

        shape = (6*self.__nbodies, self.__nmodes)
        if self.__modalDisplacements.shape != shape:
            raise ValueError(f"La taille du vecteur des déplacements modaux est incohérent : {self.__modalDisplacements.shape} VS {shape}")


    def GetNaturalFrequencies(self):
        return self.__naturalPulsations / (np.pi * 2)

    def GetNaturalPulsations(self):
        return self.__naturalPulsations

    def GetDisplacementsByMode(self):
        """Retourne une liste 'modes' au format :
            modes[k] = {"body 1" : [x, y, z, rx, ry, rz], ... }
        """
        modes = []
        for k in range(self.__nmodes) :
            mode_dict = {}
            for id_body, body in enumerate(self.__bodiesNames):
                mode_dict[body] = self.__modalDisplacements[6*id_body  : 6*(1+id_body), k]
            modes.append(mode_dict)
        return modes


    def GetDisplacementsByBodies(self):
        """Retourne une liste 'modes' au format :
            modes[body] = [[x, y, z, rx, ry, rz], # mode 1
                           [x, y, z, rx, ry, rz], # mode 2 ...
        """
        modal_results = {}
        for id_body, body in enumerate(self.__bodiesNames):
            modal_results[body] = self.__modalDisplacements[6 * id_body: 6 * (1 + id_body)]

        return modal_results


class MBSFrequencyDomainResult :
    """
    Conteneur des résultats d'analyse fréquentielle d'un système multibody.

    Cette classe stocke les fonctions de transfert calculées pour différentes
    paires entrée-sortie, ainsi que les propriétés modales du système.
    Elle fournit des méthodes d'accès et de sélection par index ou par nom.

    Attributes
    ----------
    Accessibles via méthodes Get* :
        frequency_array : np.array
            Vecteur de fréquences [Hz]
        transfer_function_array : np.array
            Fonctions de transfert complexes, shape (n_freq, n_pairs)
        natural_pulsations : np.array
            Pulsations propres du système [rad/s]
        damping_factor : np.array ou None
            Taux d'amortissement modaux (si amortissement diagonal)

    Methods
    -------
    GetFrequencyArray() : np.array
        Retourne le vecteur de fréquences [Hz]
    GetNaturalFrequencies() : np.array
        Retourne les fréquences propres [Hz]
    GetDampingFactor() : np.array ou None
        Retourne les taux d'amortissement modaux
    SelectTransferFunctionObject_byLocId(loc_index) : MBSTransferFunction
        Sélectionne une ou plusieurs FRF par indice
    SelectTransferFunctionObject_byName(loc_name) : MBSTransferFunction
        Sélectionne une ou plusieurs FRF par nom

    Examples
    --------
    >>> input_output = [["Base", 0, "Mass1", 1, ], #base -> x mass1 --> y
    >>>                 ["Base", 1, "Mass1", 1, ]] #base -> y mass1 --> y
    >>> result = system.ComputeFrequencyDomainResponse(input_output)
    >>> # Accès direct
    >>> freq = result.GetFrequencyArray()
    >>> tf_array = result.GetTransferFunctionArray()
    >>>
    >>> # Sélection par index
    >>> tf_obj = result.SelectTransferFunctionObject_byLocId(0) #base -> x mass1 --> y
    >>> plt.loglog(tf_obj.frequency, tf_obj.module)
    >>>
    >>> # Sélection par nom
    >>> tf_obj = result.SelectTransferFunctionObject_byName("Mass1::y / Base::y")
    """

    __index_to_axe = ["x", "y", "z", "θx", "θy", "θz"]
    def __init__(self, frequency_array : np.array,
                 transfer_function_array : np.array,
                 input_output_list : List,
                 natural_pulsations : np.array,
                 damping_factor : np.array = None):

        self.__input_output_list = input_output_list
        self.__frequency_array = frequency_array
        self.__transfer_function_array = transfer_function_array
        self.__naturalPulsations = natural_pulsations
        self.__dampingFactor = damping_factor

        self.__transfer_function_names = [ f"{body2}::{self.__index_to_axe[axe2]} / {body1}::{self.__index_to_axe[axe1]} "
                                           for body1,axe1,body2,axe2 in self.__input_output_list ]
        self.__transfer_function_symbols = [ f"{self.__index_to_axe[axe2]} / {self.__index_to_axe[axe1]} "
                                            for _, axe1, _, axe2 in self.__input_output_list]

        self.__nf = len(self.__transfer_function_names)

        self.__names_to_loc = {}
        for loc,name in enumerate(self.__transfer_function_names) :
            if name not in self.__names_to_loc :
                self.__names_to_loc[name] = loc

    def GetFrequencyArray(self):
        return self.__frequency_array

    def GetPulsationArray(self):
        return self.__frequency_array * 2 * np.pi

    def GetNaturalFrequencies(self):
        return self.__naturalPulsations / (np.pi * 2)

    def GetNaturalPulsations(self):
        return self.__naturalPulsations

    def GetDampingFactor(self):
        return self.__dampingFactor

    def GetTransferFunctionsNames(self):
        return self.__transfer_function_names

    def GetTransferFunctionSymbols(self):
        return self.__transfer_function_symbols

    def GetTransferFunctionArray(self):
        return self.__transfer_function_array


    def __checkSelectionLocId(self, locId: (None, int,List[int])):
        if locId is None :
            return list(range(len(self.__transfer_function_names)))
        elif  isinstance(locId, int) :
            location_id = [locId]
        elif isinstance(locId, list) and all(isinstance(i, int) for i in locId) :
            location_id = [i for i in locId]
        else :
            raise TypeError("Location index must be integer or list of integers")

        if any((loc < 0 ) or (loc + 1)> self.__nf for loc in location_id):
            raise IndexError(f"Location index is out of range of {self.__nf} transfer function elements.")

        return location_id

    def __checkSelectionLocName(self, locName: (None,str,List[str])):
        if locName is None:
            return list(range(len(self.__transfer_function_names)))
        elif isinstance(locName, str) :
            location_name = [locName]
        elif isinstance(locName, list) and all(isinstance(i, str) for i in locName) :
            location_name = [i for i in locName]
        else :
            raise TypeError("Location index must be str or list of str")

        if any(loc not in self.__transfer_function_names for loc in location_name):
            raise IndexError("Name index is not in transfer function names.")

        location_id = [self.__transfer_function_names.index(name) for name in location_name]
        return location_id

    def SelecTransferFunctionSymbols_byLocId(self, loc_index : (None, int,List[int])):
        loc_index = self.__checkSelectionLocId(loc_index)
        return [self.__transfer_function_symbols[loc] for loc in loc_index]

    def SelecTransferFunctionNames_byLocId(self, loc_index : (None, int,List[int])):
        loc_index = self.__checkSelectionLocId(loc_index)
        return [self.__transfer_function_names[loc] for loc in loc_index]

    def SelectTransferFunctionArray_byLocId(self, loc_index : (None, int,List[int])):
        loc_index = self.__checkSelectionLocId(loc_index)

        return self.__transfer_function_array[:, loc_index]

    def SelectTransferFunctionObject_byLocId(self, loc_index : (None, int,List[int])):

        return MBSTransferFunction(self.SelecTransferFunctionNames_byLocId(loc_index),
                                   self.SelecTransferFunctionSymbols_byLocId(loc_index),
                                   self.SelectTransferFunctionArray_byLocId(loc_index),
                                   self.__frequency_array)


    def SelecTransferFunctionSymbols_byName(self, loc_name : (None, str,List[str])):
        loc_index = self.__checkSelectionLocName(loc_name)
        return self.SelecTransferFunctionSymbols_byLocId(loc_index)

    def SelecTransferFunctionNames_byName(self, loc_name : (None, str,List[str])):
        loc_index = self.__checkSelectionLocName(loc_name)
        return self.SelecTransferFunctionNames_byLocId(loc_index)

    def SelectTransferFunctionArray_byName(self, loc_name : (None, str,List[str])):
        loc_index = self.__checkSelectionLocName(loc_name)
        return self.SelectTransferFunctionArray_byLocId(loc_index)

    def SelectTransferFunctionObject_byName(self, loc_name : (None, str,List[str])):
        loc_index = self.__checkSelectionLocName(loc_name)
        return self.SelectTransferFunctionObject_byLocId(loc_index)




class MBSTransferFunction :
    """
    Représentation d'une ou plusieurs fonctions de transfert.

    Cette classe encapsule les données d'une fonction de transfert complexe
    et fournit des propriétés calculées (module, phase, parties réelle/imaginaire, PSD).

    Attributes (accessibles via @property)
    ----------
    names : List[str]
        Noms complets des FRF (ex: "Mass1::y / Base::x")
    symbols : List[str]
        Symboles abrégés des FRF (ex: "y / x")
    transferFunction : np.array
        Valeurs complexes G(ω), shape (n_freq, n_tf)
    module : np.array
        Module |G(ω)|, shape (n_freq, n_tf)
    phase : np.array
        Phase arg(G(ω)) en degrés, dépliée, shape (n_freq, n_tf)
    real : np.array
        Partie réelle Re(G(ω)), shape (n_freq, n_tf)
    imaginary : np.array
        Partie imaginaire Im(G(ω)), shape (n_freq, n_tf)
    frequency : np.array
        Fréquences [Hz], shape (n_freq,)
    pulsation : np.array
        Pulsations [rad/s], shape (n_freq,)
    powerSpectralDensity : np.array
        PSD = |G(ω)|², shape (n_freq, n_tf)

    Examples
    --------
    >>> tf = result.SelectTransferFunctionObject_byLocId(0)
    >>>
    >>> # Diagramme de Bode
    >>> fig, (ax1, ax2) = plt.subplots(2, 1)
    >>> ax1.loglog(tf.frequency, tf.module)
    >>> ax1.set_ylabel('Module')
    >>> ax2.semilogx(tf.frequency, tf.phase)
    >>> ax2.set_ylabel('Phase [°]')
    >>> ax2.set_xlabel('Fréquence [Hz]')
    >>>
    >>> # Diagramme de Nyquist
    >>> plt.plot(tf.real[:, 0], tf.imaginary[:, 0])
    >>> plt.axis('equal')
    >>> plt.xlabel('Re(G)')
    >>> plt.ylabel('Im(G)')
    """

    def __init__(self,tf_names : List[str], tf_symbols : List[str], tf_array : np.array, freq_array : np.array):
        self.__names = tf_names
        self.__symbols = tf_symbols
        self.__tf_array = tf_array
        self.__freq_array = freq_array

    @property
    def names(self):
        return self.__names

    @property
    def symbols(self):
        return self.__symbols

    @property
    def transferFunction(self):
        return self.__tf_array

    @property
    def module(self):
        return np.abs(self.__tf_array)

    @property
    def phase(self):
        # Calcul de la phase brute en radians
        phase_rad = np.angle(self.__tf_array)

        # Dépliage pour chaque fonction de transfert (par colonne)
        if phase_rad.ndim == 1:
            # Cas d'une seule FRF
            phase_unwrapped = np.unwrap(phase_rad)
        else:
            # Cas de plusieurs FRF
            phase_unwrapped = np.unwrap(phase_rad, axis=0)

        # Conversion en degrés
        phase_deg = np.degrees(phase_unwrapped)
        return phase_deg - phase_deg[0]

    @property
    def frequency(self):
        return self.__freq_array

    @property
    def pulsation(self):
        return self.__freq_array * 2 * np.pi

    @property
    def imaginary(self):
        return np.imag(self.__tf_array)

    @property
    def real(self):
        return np.real(self.__tf_array)

    @property
    def powerSpectralDensity(self):
        """
            Calcule la densité spectrale de puissance (PSD) des fonctions de transfert.

            La PSD est définie comme le carré du module : PSD(ω) = |G(ω)|²
            Elle représente la distribution d'énergie en fonction de la fréquence.

            Returns
            -------
            np.array
                PSD en unités au carré, shape (n_freq, n_tf)

            Notes
            -----
            Pour obtenir la PSD d'un signal de sortie x(t) à partir d'un signal
            d'entrée u(t) de PSD connue Sᵤᵤ(ω), utiliser :
                Sₓₓ(ω) = |G(ω)|² × Sᵤᵤ(ω)

            Unités typiques :
            - Si G est sans dimension : PSD sans dimension
            - Si G en [m/m] : PSD sans dimension
            - Si G en [m/N] : PSD en [m²/N²]
            """
        return np.abs(self.__tf_array) ** 2

