import numpy as np
from typing import List

from MultiBodySimulation.MBSBody import MBSRigidBody3D
from MultiBodySimulation.MBS_numerics import RotationMatrix

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


    def get_connected_point_motion(self,global_point):
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
        self.__naturalFrequencies = natural_pulsations / (np.pi * 2)
        self.__modalDisplacements = modal_displacements
        self.__bodiesNames = body_names_list

        self.__nmodes = len(natural_pulsations)
        self.__nbodies = len(body_names_list)

        shape = (6*self.__nbodies, self.__nmodes)
        if self.__modalDisplacements.shape != shape:
            raise ValueError(f"La taille du vecteur des déplacements modaux est incohérent : {self.__modalDisplacements.shape} VS {shape}")


    def GetNaturalFrequencies(self):
        return self.__naturalFrequencies

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