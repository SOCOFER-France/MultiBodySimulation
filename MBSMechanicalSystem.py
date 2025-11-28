import numpy as np
from typing import Dict
from scipy.sparse import csc_matrix
from scipy.integrate import solve_ivp

from MultiBodySimulation.MBSMechanicalJoint import _MBSLink3D
from MultiBodySimulation.MBSBody import MBSRigidBody3D
from MultiBodySimulation.MBSSimulationResults import MBSBodySimulationResult

class MBSMechanicalSystem3D:
    def __init__(self):
        self.__assembled = False
        self.__mechaConnectors = False
        self.bodies = []
        self.body_index = {}

        self.ref_bodies = []
        self.ref_body_index = {}

        self.links = []

        self.__nrefbodies = 0
        self.__nbodies = 0

        self.gravity = np.array([0.,0.,0.])

        self.__bodies_dof = []
        self.__freedof = []
        self.__fixeddof = []

        # matrice de raideur
        self.__Kmatrix = None
        # matrice d'amortissement visqueux
        self.__Cmatrix = None
        # matrice de masse
        self.__Mmatrix = None

        self.__Mff = None
        self.__invMff = None
        self.__Kff = None
        self.__Cff = None
        self.__Kb = None
        self.__Cb = None

        self.__nbodies = 0
        self.__nrefbodies = 0
        self.__ntot = 0

        self.__Jac_linear = None

    @property
    def __allbodies(self):
        return self.ref_bodies + self.bodies

    def _block_slice(self, body_idx: int) -> slice:
        """Renvoie slice pour le bloc 6x6 du corps i."""
        s = 6*body_idx
        return slice(s, s+6)

    def _index_bodies(self):
        """Crée un mapping corps -> bloc index (start pos in global matrices)."""
        self.body_index_map: Dict[object, int] = {}
        for i, b in enumerate(self.__allbodies):
            self.body_index_map[b] = i


        self.__x_indices = []
        self.__theta_indices = []
        for i in range(self.__nbodies):
            s = 6*i
            self.__x_indices += [s, s+1, s+2]
            self.__theta_indices += [s+3, s+3 + 1, s+3 + 2]


        self.__xref_indices = []
        self.__thetaref_indices = []
        for i in range(self.__nrefbodies):
            s = 6*i
            self.__xref_indices += [s, s+1, s+2]
            self.__thetaref_indices += [s+3, s+3 + 1, s+3 + 2]


    def __vecProductMatrix(self,A,B):
        x = (B-A)
        xi,yi,zi = x


        mat = np.array([[0,-zi,yi],
                        [zi,0,-xi],
                        [-yi,xi,0]] )
        return mat



    def GetBodyByName(self,name):
        idBody = self.body_index.get(name,None)
        if idBody is not None :
            return self.bodies[idBody]
        idBody = self.ref_body_index.get(name,None)
        if idBody is not None :
            return self.ref_bodies[idBody]

        raise IndexError(f"No body named : '{name}' in the system.")

    def AddRigidBody(self, body : MBSRigidBody3D):
        if self.__assembled :
            raise ValueError("Cannot connect new body while the system is already assembled.")
        if body.IsFixed :
            if body.GetName in self.body_index or body.GetName in self.ref_body_index:
                raise ValueError(f"Body {body.GetName} already exists.")
            self.ref_body_index[body.GetName] = len(self.ref_bodies)
            self.ref_bodies.append(body)
            self.__nrefbodies = len(self.ref_bodies)

        else :
            if body.GetName in self.body_index or body.GetName in self.ref_body_index:
                raise ValueError(f"Body {body.GetName} already exists.")
            self.body_index[body.GetName] = len(self.bodies)
            self.bodies.append(body)
            self.__nbodies = len(self.bodies)

    def AddLinkage(self, link : _MBSLink3D):
        if self.__assembled :
            raise ValueError("Cannot connect new link while the system is already assembled.")
        self.links.append(link)


    def _get_refbodies_displacement_state(self, t):
        y = []
        dydt = []
        for body in self.ref_bodies :
            Dy = body._updateDisplacement(t)

            y.append(Dy[:6])
            dydt.append(Dy[6:])
        return np.concatenate(y + dydt)

    def _get_bodies_initial_displacement(self):
        y = []
        dydt = []
        for body in self.bodies :
            y.append(body._initial_position - body._referencePosition)
            y.append(body._initial_angles - body._refAngles)
            dydt.append(body._velocity)
            dydt.append(body._omega)
        return np.concatenate(y + dydt)

    def _get_refbodies_position_state(self, t):
        y = []
        dydt = []
        for body in self.ref_bodies:
            Dy = body._updateDisplacement(t)
            Dy[:3] += body._referencePosition
            Dy[3:6:] += body._refAngles
            y.append(Dy[:6])
            dydt.append(Dy[6:])
        return np.concatenate(y + dydt)

    def _get_bodies_reference_position(self):
        y = []
        dydt = []
        for body in self.bodies :
            y.append(body._referencePosition)
            y.append(body._refAngles)
            dydt.append([0.] * 12)
        return np.concatenate(y + dydt)

    def _recompose_ref_body_position(self, Dy):
        y = Dy.copy()
        for i, body in enumerate(self.ref_bodies):
            y[6 * i:6 * i + 3] = body._referencePosition[:, None] + Dy[6 * i:6 * i + 3]
            y[6 * i + 3:6 * i + 6] = body._refAngles[:, None] + Dy[6 * i + 3:6 * i + 6]
        return y

    def _recompose_body_position(self, Dy):
        y = Dy.copy()
        for i, body in enumerate(self.bodies):
            y[6 * i:6 * i + 3] = body._referencePosition[:, None] + Dy[6 * i:6 * i + 3]
            y[6 * i + 3:6 * i + 6] = body._refAngles[:, None] + Dy[6 * i + 3:6 * i + 6]
        return y



    def _assemble_matrices(self):
        nbodies = len(self.__allbodies)
        # matrice de raideur
        self.__Kmatrix = np.zeros((6*nbodies, 6*nbodies), dtype=float)
        # matrice d'amortissement visqueux
        self.__Cmatrix = np.zeros((6*nbodies, 6*nbodies), dtype=float)
        # matrice de masse
        self.__Mmatrix = np.zeros((6*nbodies, 6*nbodies), dtype=float)
        # Matrice Q et P
        nlinks = len(self.links)
        self.__Qmat_linkage = np.zeros((6 * nlinks, 6 * nbodies))
        self.__Pmat_linkage = np.zeros((6 * nbodies, 6 * nlinks))
        self.__Kmat_linkage = np.zeros((6 * nlinks, 6 * nlinks))
        self.__Cmat_linkage = np.zeros((6 * nlinks, 6 * nlinks))

        self.__nbodies = len(self.bodies)
        self.__nrefbodies = len(self.ref_bodies)
        self.__ntot = self.__nbodies + self.__nrefbodies

        self.__freedof = []
        self.__fixeddof = []

        self.__gravity_matrix = np.tile(np.concatenate([self.gravity,[0,0,0]]),self.__nbodies)

        for k, body_k in enumerate(self.__allbodies):
            s = self._block_slice(k)
            dof = list(range(s.start,s.start+6))
            if not body_k.IsFixed :

                self.__Mmatrix[s.start:s.start+3][:, s.start:s.start+3] = np.eye(3) * body_k._mass
                self.__Mmatrix[s.start+3:s.start+6][:, s.start+3:s.start+6] =  body_k._inertia

                self.__freedof += dof
            else :
                self.__fixeddof += dof

        # Liaisons non-linéaires
        self.__non_linear_link = []
        self.__linear_link = []

        # Liaisons contraintes
        nconstraints = len([i for i,l in enumerate(self.links) if l.IsKinematic])
        Pcons = np.zeros((nbodies * 6, 6 * nconstraints))
        Qcons = np.zeros((nconstraints * 6, 6 * nbodies))
        Kcons = np.zeros((nconstraints * 6, nconstraints * 6))
        Ccons = np.zeros((nconstraints * 6, nconstraints * 6))
        id_constraint = 0

        # Liaisons avec jeu
        ngap = len([i for i,l in enumerate(self.links) if l.HasGap])
        self.__n_gapLink = ngap
        stop_delta_plus = np.ones(ngap * 6, dtype=float) * np.inf
        stop_delta_minus = -np.ones(ngap * 6, dtype=float) * np.inf
        Pgap = np.zeros((nbodies * 6, 6 * ngap))
        Qgap = np.zeros((ngap * 6, 6 * nbodies))
        Kgap = np.zeros((ngap * 6, ngap * 6))
        Cgap = np.zeros((ngap * 6, ngap * 6))
        id_gap = 0


        for id_link, link in enumerate(self.links) :

            b1 = link.GetBody1
            b2 = link.GetBody2
            i = self.body_index_map[b1]
            j = self.body_index_map[b2]
            si = self._block_slice(i)
            sj = self._block_slice(j)

            G1 = b1.GetReferencePosition()
            G2 = b2.GetReferencePosition()

            O1 = link.GetGlobalPoint1
            O2 = link.GetGlobalPoint2

            A1 = np.eye(6)
            A2 = np.eye(6)


            D_G1_O = self.__vecProductMatrix(G1, O1)
            D_G2_O = self.__vecProductMatrix(G2, O2)


            A1[3:][:, :3] = D_G1_O
            A2[3:][:, :3] = D_G2_O

            B1 = A1.T
            B2 = A2.T

            # Partie linéaire des réactions
            Kt,Ct, Kth,Cth = link.GetLinearReactionMatrices

            # linéaire / non-linéaire
            link_prop = (link, si, sj, A1,A2, B1,B2)
            if link.IsLinear :
                self.__linear_link.append(link_prop)
            else :
                self.__non_linear_link.append(link_prop)


            # matrices locales 6x6 liant déplacement point -> effort point
            Kloc = np.zeros((6, 6))
            Cloc = np.zeros((6, 6))

            Kloc[0:3][:, 0:3] = Kt
            Kloc[3:6][:, 3:6] = Kth
            Cloc[0:3][:, 0:3] = Ct
            Cloc[3:6][:, 3:6] = Cth


            if link.IsKinematic :
                s_cons = slice(id_constraint * 6, id_constraint * 6 + 6)
                Qcons[s_cons, si] = B1
                Qcons[s_cons, sj] = -B2

                Pcons[si, s_cons] = A1
                Pcons[sj, s_cons] = -A2  # Avec A1 + et A2 - ==> P @ K @ Q === Kmat

                Kcons[s_cons, s_cons] = Kloc
                Ccons[s_cons, s_cons] = Cloc


            if link.HasGap :
                s_gap = slice(id_gap * 6, id_gap*6 + 6)
                s_gap_trans = slice(id_gap * 6, id_gap*6 + 3)
                s_gap_rot = slice(id_gap * 6 + 3, id_gap * 6 + 6)
                Qgap[s_gap, si] = -B1
                Qgap[s_gap, sj] = B2

                Pgap[si, s_gap] = -A1
                Pgap[sj, s_gap] = A2  # Avec A1/B1 - et A2/B2 + ==> P @ K @ Q === Kmat

                Kgap[s_gap, s_gap] = Kloc
                Cgap[s_gap, s_gap] = Cloc



                stop_delta_plus[s_gap_trans] = link.GetTransGap[:,1]
                stop_delta_plus[s_gap_rot] = link.GetRotGap[:, 1]
                stop_delta_minus[s_gap_trans] = link.GetTransGap[:, 0]
                stop_delta_minus[s_gap_rot] = link.GetRotGap[:, 0]

                continue

            s_linkage = slice(id_link * 6, id_link * 6 + 6)
            # Avec A1/B1 - et A2/B2 + ==> P @ K @ Q === Kmat
            self.__Qmat_linkage[s_linkage][:, si] += -B1
            self.__Qmat_linkage[s_linkage][:, sj] += B2

            self.__Pmat_linkage[si][:, s_linkage] += -A1
            self.__Pmat_linkage[sj][:, s_linkage] += A2

            self.__Kmat_linkage[s_linkage][:, s_linkage] += Kloc
            self.__Cmat_linkage[s_linkage][:, s_linkage] += Cloc

            # contribution aux matrices globales : T_i^T * Kloc * T_i, etc.

            # K11 = A1.dot(Kloc).dot(B1)
            # K12 = -A1.dot(Kloc).dot(B2)
            # K22 = A2.dot(Kloc).dot(B2)
            # K21 = -A2.dot(Kloc).dot(B1)
            #
            # C11 = A1.dot(Cloc).dot(B1)
            # C12 = -A1.dot(Cloc).dot(B2)
            # C22 = A2.dot(Cloc).dot(B2)
            # C21 = -A2.dot(Cloc).dot(B1)
            #
            #
            # # assemble in global matrices
            # self.__Kmatrix[si][:, si] += K11
            # self.__Kmatrix[sj][:, sj] += K22
            # self.__Kmatrix[si][:, sj] += K12
            # self.__Kmatrix[sj][:, si] += K21
            #
            # self.__Cmatrix[si][:, si] += C11
            # self.__Cmatrix[sj][:, sj] += C22
            # self.__Cmatrix[si][:, sj] += C12
            # self.__Cmatrix[sj][:, si] += C21




        keepgap = (~np.isinf(stop_delta_plus)) & (~np.isinf(stop_delta_minus))
        self.__Pmat_gap = Pgap[:, keepgap]
        self.__Qmat_gap = Qgap[keepgap]
        self.__Kmat_gap = Kgap[keepgap][:, keepgap]
        self.__Cmat_gap = Cgap[keepgap][:, keepgap]
        self.__gapPlus = stop_delta_plus[keepgap]
        self.__gapMinus = stop_delta_minus[keepgap]


        self.__Kmatrix = self.__Pmat_linkage @ self.__Kmat_linkage @ self.__Qmat_linkage
        self.__Cmatrix = self.__Pmat_linkage @ self.__Cmat_linkage @ self.__Qmat_linkage

    def _partition_matrices(self):
        """Partitionne M,C,K et F en sous-systèmes libres/prescrits :
                   Retourne (invMff, Cff, Kff, Kfb, Cfb)
                """
        if self.__Kmatrix is None : return

        self.__Mff = self.__Mmatrix[self.__freedof][:,self.__freedof]
        self.__invMff = np.linalg.inv(self.__Mff)
        self.__Kff = self.__Kmatrix[self.__freedof][:,self.__freedof]
        self.__Cff = self.__Cmatrix[self.__freedof][:,self.__freedof]
        self.__Kb = self.__Kmatrix[self.__freedof][:,self.__fixeddof]
        self.__Cb = self.__Cmatrix[self.__freedof][:,self.__fixeddof]

        self.__Pgap_f = self.__Pmat_gap[self.__freedof]
        self.__Pgap_b = self.__Pmat_gap[self.__fixeddof]
        self.__Qgap_f = self.__Qmat_gap[:,self.__freedof]
        self.__Qgap_b = self.__Qmat_gap[:,self.__fixeddof]
        return



    def CheckInitialTensions(self, t0):
        if not self.__assembled :
            raise ValueError("Système non assemblé")

        y0_fixed = self._get_refbodies_displacement_state(t0)
        y0 = self._get_bodies_initial_displacement()  # vecteur déplacement / vitesse (dX, v, dTheta, omega)

        Ufixed = y0_fixed[:6 * self.__nrefbodies]
        Vfixed = y0_fixed[6 * self.__nrefbodies:]

        Uvec = y0[:6 * self.__nbodies]
        Vvec = y0[6 * self.__nbodies:]

        U = np.zeros(6 * self.__ntot, dtype=float)
        V = np.zeros_like(U, dtype=float)
        U[self.__fixeddof] = Ufixed
        U[self.__freedof] = Uvec
        V[self.__fixeddof] = Vfixed
        V[self.__freedof] = Vvec

        init_tension = []
        for (link, si, sj, A1, A2, B1, B2) in self.__non_linear_link + self.__linear_link :
            b1 = link.GetBody1
            b2 = link.GetBody2
            i = self.body_index_map[b1]
            j = self.body_index_map[b2]
            si = self._block_slice(i)
            sj = self._block_slice(j)

            Ui = U[si]
            Vi = V[si]
            Uj = U[sj]
            Vj = V[sj]

            Ui_point = B1 @ Ui
            Vi_point = B1 @ Vi
            Uj_point = B2 @ Uj
            Vj_point = B2 @ Vj


            if link.IsLinear :
                f = link.GetLinearLocalReactions(Ui_point, Vi_point, Uj_point, Vj_point)
            else :
                f = link.GetNonLinearLocalReactions(Ui_point, Vi_point, Uj_point, Vj_point)

            if (np.abs(f)>0).any() :
                init_tension.append( (b1.GetName, b2.GetName, np.round(f[:6],5) ) )
        if len(init_tension)>0:
            print("Warning : tension initiale dans des liaisons.")
        for (b1,b2,f) in init_tension :
            print(b1,">>>",b2," : {force | torque}\n", f)

    def _non_linear_forces(self,y, yfixed):
        U = np.zeros(6 * self.__ntot, dtype=float)
        V = np.zeros_like(U, dtype=float)

        U[self.__fixeddof] = yfixed[:6 * self.__nrefbodies]
        V[self.__fixeddof] = yfixed[6 * self.__nrefbodies:]
        U[self.__freedof] = y[:6 * self.__nbodies]
        V[self.__freedof] = y[6 * self.__nbodies:]

        F = np.zeros(self.__ntot*6, dtype=float)
        for (link, si, sj, A1,A2, B1,B2) in self.__non_linear_link:

            # extraire U1, V1, U2, V2 (local point displacements)
            Ui = U[si]
            Vi = V[si]
            Uj = U[sj]
            Vj = V[sj]
            Ui_point = B1 @ Ui
            Vi_point = B1 @ Vi
            Uj_point = B2 @ Uj
            Vj_point = B2 @ Vj

            # appel à la méthode spécifique de la liaison
            F_torque_point = link.GetNonLinearLocalReactions(Ui_point, Vi_point, Uj_point, Vj_point)
            # F_torque_point est un vecteur 6 (force, torque) appliqué au point (sur body1), opposite on body2
            # redispatch to CDG

            F_cdg1 = A1 @ np.asarray(F_torque_point)
            F_cdg2 = - A2 @ np.asarray(F_torque_point)

            F[si] += F_cdg1
            F[sj] += F_cdg2
        return F[self.__freedof]


    def AssemblyMatrixSystem(self):
        self._index_bodies()
        self._assemble_matrices()
        self._partition_matrices()
        self.__assembled = True


    def RunDynamicSimulation(self, t_span, dt, ode_method="BDF", print_step_rate=0,
                             smallAngles = True , smallAnglesDegreesThreshold=5.0):
        if not self.__assembled :
            self.AssemblyMatrixSystem()

        if not smallAngles :
            smallAnglesThreshold = smallAnglesDegreesThreshold / 180 * np.pi
        else :
            smallAnglesThreshold = None

        nt = int( np.ceil( (t_span[1] - t_span[0]) / dt ) + 1 )
        t_eval = np.linspace(t_span[0], t_span[1], nt)
        new_dt = t_eval[1] - t_eval[0]
        if not np.isclose(new_dt, dt) :
            print(f"Recalcule de dt >> {dt:.4e}  --> {new_dt:.4e}")


        if ode_method in ["BDF", "Radau"] :
            y, yfixed, Dy, Dyfixed = self._RunSolveIVP(t_eval,
                                                       print_step_rate,
                                                       method = ode_method,
                                                       smallAnglesThreshold = smallAnglesThreshold)

        result_dict = {}

        for body in self.ref_bodies:
            idx = self.ref_body_index[body.GetName]
            Dy_body = Dyfixed[6 * idx : 6 * (idx + 1)] # Dx,angle,
            y_body = yfixed[6 * idx : 6 * (idx + 1)] # Positions, angle
            v_body = Dyfixed[12*self.__nrefbodies + 6 * idx: 12*self.__nrefbodies + 6 * (idx+1)] # V omega
            result_dict[body.GetName] = MBSBodySimulationResult(body,
                                                             t_eval,
                                                             Dy_body,
                                                             y_body,
                                                             v_body)

        for body in self.bodies:
            idx = self.body_index[body.GetName]
            Dy_body = Dy[6 * idx: 6 * (idx + 1)]  # Dx,angle,
            y_body = y[6 * idx: 6 * (idx + 1)]  # Positions, angle
            v_body = Dy[12 * self.__nbodies + 6 * idx: 12 * self.__nbodies + 6 * (idx + 1)]  # V omega
            result_dict[body.GetName] = MBSBodySimulationResult(body,
                                                                t_eval,
                                                                Dy_body,
                                                                y_body,
                                                                v_body)
        return t_eval, result_dict



    def _RunSolveIVP(self, t_eval,
                     print_step_rate : int =0,
                     method : str = "BDF",
                     smallAnglesThreshold = None):
        if not isinstance(print_step_rate, int) :
            raise TypeError("'print_step_rate' must be integer. 0 to disable printing.")


        nt = len(t_eval)
        t0, Dy0 = self.__initialState(t_eval)
        y0 = self._get_bodies_reference_position()
        yfixed0 = self._get_refbodies_position_state(0)
        U = np.zeros(self.__ntot*6)
        U[self.__freedof] = y0[:self.__nbodies*6]
        U[self.__fixeddof] = yfixed0[:self.__nrefbodies * 6]
        du0 = self.__Qmat_linkage @ U
        print(du0)
        Du = np.zeros_like(U)
        Dyfixed0 = self._get_refbodies_displacement_state(0)
        Du[self.__freedof] = Dy0[:self.__nbodies * 6]
        Du[self.__fixeddof] = Dyfixed0[:self.__nrefbodies * 6]
        du0 = self.__Qmat_linkage @ Du
        print(du0)

        if print_step_rate <= 1 :
            substep = 1
            steps = np.array([0,nt],dtype=int)
        else :
            steps = np.unique([int(s) for s in np.linspace(0,nt,print_step_rate)])
            substep = len(steps) - 1

        Dy = np.zeros((self.__nbodies * 12, nt))
        Dyfixed = np.zeros((self.__nrefbodies * 12, nt))

        # Notes :
        # Dy << vecteur état déplacements des corps libres 12 composantes
        # Dx déplacement / déformation translation
        # Dtheta déformation angulaire
        # Vx vitesse en translation
        # omega vitesse angulaire
        # structure Dy = { [Dx, Dtheta]_0,
        #                  [Dx, Dtheta]_1,
        #                   ...
        #                  [Dx, Dtheta]_n,
        #                  [Vx, omega]_0,
        #                   ...
        #                  [Vx, omega]_n,}


        if self.__n_gapLink == 0 :
            jac = self._approxJacobian()
        else :
            jac = self._approxJacobian


        for k, (start_substep, end_substep) in enumerate(zip(steps[:-1],steps[1:]),start=1) :
            t_substep = t_eval[start_substep:end_substep:]
            t_span = (t_substep[0], t_substep[-1])

            sol = solve_ivp(self.__IVP_derivativeFunc,
                            t_span,
                            Dy0,
                            method=method,
                            t_eval=t_substep,
                            jac=jac,
                            )

            Dyfixed[:,start_substep:end_substep-1:] = np.array([self._get_refbodies_displacement_state(ti) for ti in t_substep[:-1]]).T
            Dy[:, start_substep:end_substep-1:] = sol.y[:,:-1]

            Dy0 = sol.y[:,-1]
            if k == substep :
                # Dernière itération
                Dy[:, -1] = Dy0
                Dyfixed[:, -1] = self._get_refbodies_displacement_state(t_substep[-1])

            print("Step X / N ... blablabla")

        y = self._recompose_body_position(Dy)
        yfixed = self._recompose_ref_body_position(Dyfixed)

        # Notes :
        # y << vecteur position corps libres 12 composantes
        # x position
        # theta angle d'euler
        # Vx vitesse
        # omega vitesse angulaire
        return y, yfixed, Dy, Dyfixed


    def __initialState(self,t_eval):
        t0 = t_eval[0]
        y0_fixed = self._get_refbodies_displacement_state(t0)
        y0 = self._get_bodies_initial_displacement() # vecteur déplacement / vitesse (dX, v, dTheta, omega)


        Ufixed = y0_fixed[:6 * self.__nrefbodies]
        Vfixed = y0_fixed[6 * self.__nrefbodies:]

        Uvec = y0[:6 * self.__nbodies]
        Vvec = y0[6 * self.__nbodies:]

        U = np.zeros(6 * self.__ntot, dtype=float)
        V = np.zeros_like(U, dtype=float)
        U[self.__fixeddof] = Ufixed
        U[self.__freedof] = Uvec
        V[self.__fixeddof] = Vfixed
        V[self.__freedof] = Vvec

        return t0, y0

    def _nonLinearForces(self, y, yfixed):
        U = np.zeros(6 * self.__ntot, dtype=float)
        V = np.zeros_like(U, dtype=float)

        U[self.__fixeddof] = yfixed[:6 * self.__nrefbodies]
        V[self.__fixeddof] = yfixed[6 * self.__nrefbodies:]
        U[self.__freedof] = y[:6 * self.__nbodies]
        V[self.__freedof] = y[6 * self.__nbodies:]

        F = np.zeros(self.__ntot * 6, dtype=float)
        for (link, si, sj, A1, A2, B1, B2) in self.__non_linear_link:
            # extraire U1, V1, U2, V2 (local point displacements)
            Ui = U[si]
            Vi = V[si]
            Uj = U[sj]
            Vj = V[sj]
            Ui_point = B1 @ Ui
            Vi_point = B1 @ Vi
            Uj_point = B2 @ Uj
            Vj_point = B2 @ Vj

            # appel à la méthode spécifique de la liaison
            F_torque_point = link.GetNonLinearLocalReactions(Ui_point, Vi_point, Uj_point, Vj_point)
            # F_torque_point est un vecteur 6 (force, torque) appliqué au point (sur body1), opposite on body2
            # redispatch to CDG

            F_cdg1 = A1 @ np.asarray(F_torque_point)
            F_cdg2 = - A2 @ np.asarray(F_torque_point)

            F[si] += F_cdg1
            F[sj] += F_cdg2
        return F[self.__freedof]


    def _penalizedGapContactReactionForces(self,y , yfixed):
        ub = yfixed[:6 * self.__nrefbodies]
        vb = yfixed[6 * self.__nrefbodies:]
        u = y[:6 * self.__nbodies]
        v = y[6 * self.__nbodies:]


        du = (self.__Qgap_f @ u + self.__Qgap_b @ ub)
        dv = (self.__Qgap_f @ v + self.__Qgap_b @ vb)
        du_viol = np.maximum(0., du - self.__gapPlus) +\
                  np.minimum(0., du - self.__gapMinus)
        dv_viol = dv * (np.abs(du_viol) > 0.)

        F = -self.__Pgap_f @ (self.__Kmat_gap @ du_viol + self.__Cmat_gap @ dv_viol)

        return F

    def __IVP_derivativeFunc(self, t, y):
        yfixed = self._get_refbodies_displacement_state(t)
        Ufixed = yfixed[:6 * self.__nrefbodies]
        Vfixed = yfixed[6 * self.__nrefbodies:]

        Uvec = y[:6 * self.__nbodies]
        Vvec = y[6 * self.__nbodies:]

        # Forces de réaction visco-élastiques linéaires
        F = -(self.__Kff @ Uvec + self.__Cff @ Vvec + self.__Kb @ Ufixed + self.__Cb @ Vfixed)

        if len(self.__non_linear_link)>0 :
            F += self._nonLinearForces(y, yfixed)
        if self.__n_gapLink > 0 :
            F += self._penalizedGapContactReactionForces(y, yfixed)
        acc = self.__invMff @ F + self.__gravity_matrix

        return np.concatenate([Vvec, acc])

    def _approxJacobian(self, t=None, y=None):
        if self.__Jac_linear is None :
            n = 6 * self.__nbodies
            A = np.zeros((2 * n, 2 * n))
            A[:n, n:] = np.eye(n)
            A[n:, :n] = -self.__invMff @ self.__Kff
            A[n:, n:] = -self.__invMff @ self.__Cff
            self.__Jac_linear = csc_matrix(A)

        if self.__n_gapLink == 0 :
            return self.__Jac_linear

        n = 6 * self.__nbodies
        u = y[:n]

        yref = self._get_refbodies_displacement_state(t)
        ub = yref[:6 * self.__nrefbodies]

        du = (self.__Qgap_f @ u + self.__Qgap_b @ ub)

        phi0 = np.maximum(0, du - self.__gapPlus) + np.minimum(0, du - self.__gapMinus)
        s_phi = 1.0 * (phi0 > 0) + 1.0 * (phi0 < 0)
        Pf = (self.__Pgap_f * s_phi[np.newaxis])

        Agap_penal = np.zeros((2 * n, 2 * n))
        Agap_penal[n:, :n] = -self.__invMff @ (Pf @ ((self.__Kmat_gap @ self.__Qgap_f)))
        Agap_penal[n:, n:] = -self.__invMff @ (Pf @ ((self.__Cmat_gap @ self.__Qgap_f)))
        Agap_penal = csc_matrix(Agap_penal)
        return Agap_penal + self.__Jac_linear


