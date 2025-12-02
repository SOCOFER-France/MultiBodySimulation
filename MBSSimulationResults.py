import numpy as np

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