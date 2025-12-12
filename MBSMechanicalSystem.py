import numpy as np
from typing import Dict, List, Optional
from scipy.sparse import csc_matrix
from scipy.integrate import solve_ivp
from scipy.linalg import eigvalsh, eigh
from scipy.linalg import qr as QR_decomposition
import warnings

from MultiBodySimulation.MBSMechanicalJoint import _MBSLink3D
from MultiBodySimulation.MBSBody import MBSRigidBody3D, MBSReferenceBody3D
from MultiBodySimulation.MBSSimulationResults import (MBSBodySimulationResult,
                                                      MBSModalResults,
                                                      MBSFrequencyDomainResult)



class __MBSBase:
    """
    Classe de base pour les systèmes multicorps.

    Gère la structure du système (corps, liaisons) et l'assemblage des matrices.
    Les classes dérivées implémentent les méthodes de résolution spécifiques.
    """

    def __init__(self):


        # Corps et liaisons
        self.bodies: List[MBSRigidBody3D] = []
        self.body_index: Dict[str, int] = {}
        self.ref_bodies: List[MBSReferenceBody3D] = []
        self.ref_body_index: Dict[str, int] = {}
        self.links: List[_MBSLink3D] = []

        # Compteurs
        self._nrefbodies = 0
        self._nbodies = 0
        self._ntot = 0
        self._nlinks = 0

        # Gravité
        self.gravity = np.array([0., 0., 0.])

        # DOF
        self._freedof = []
        self._fixeddof = []



        # Mapping corps → indices matrices
        self.body_index_map: Dict[object, int] = {}

    @property
    def _allbodies(self):
        """Tous les corps (fixes + libres)"""
        return self.ref_bodies + self.bodies

    def AddRigidBody(self, body: MBSRigidBody3D):
        """Ajoute un corps rigide au système."""
        if self._assembled:
            raise ValueError("Cannot add body while system is assembled.")

        if body.IsFixed:
            if body.GetName in self.body_index or body.GetName in self.ref_body_index:
                raise ValueError(f"Body {body.GetName} already exists.")
            self.ref_body_index[body.GetName] = len(self.ref_bodies)
            self.ref_bodies.append(body)
            self._nrefbodies = len(self.ref_bodies)
        else:
            if body.GetName in self.body_index or body.GetName in self.ref_body_index:
                raise ValueError(f"Body {body.GetName} already exists.")
            self.body_index[body.GetName] = len(self.bodies)
            self.bodies.append(body)
            self._nbodies = len(self.bodies)

    def AddLinkage(self, link: _MBSLink3D):
        """Ajoute une liaison mécanique au système."""
        if self._assembled:
            raise ValueError("Cannot add linkage while system is assembled.")
        self.links.append(link)

    def GetBodyByName(self, name: str) -> MBSRigidBody3D:
        """Récupère un corps par son nom."""
        idBody = self.body_index.get(name, None)
        if idBody is not None:
            return self.bodies[idBody]
        idBody = self.ref_body_index.get(name, None)
        if idBody is not None:
            return self.ref_bodies[idBody]
        raise IndexError(f"No body named: '{name}' in the system.")

    def _get_fixedBodies_displacement_state(self, t: float) -> np.ndarray:
        """
        Calcule le vecteur déplacement des corps fixes (dynamique imposée) à l'instant t.

        Returns:
            Array [Δu₁, Δθ₁, ..., Δuₙ, Δθₙ, v₁, ω₁, ..., vₙ, ωₙ]
        """
        y = []
        dydt = []
        for body in self.ref_bodies:
            Dy = body._updateDisplacement(t)

            y.append(Dy[:6])  # Déplacements [Δx, Δθ]
            dydt.append(Dy[6:])  # Vitesses [v, ω]
        return np.concatenate(y + dydt)

    def _get_fixedBodies_position_state(self, t: float) -> np.ndarray:
        """
        Retourne les positions absolues des corps fixes à l'instant t.

        Returns:
            Array [x₁, θ₁, ..., xₙ, θₙ, v₁, ω₁, ..., vₙ, ωₙ]
        """
        y = []
        dydt = []
        for body in self.ref_bodies:
            Dy = body._updateDisplacement(t)
            # Position absolue = référence + déplacement
            y.append(Dy[:3] + body._referencePosition)
            y.append(Dy[3:6] + body._refAngles)
            dydt.append(Dy[6:])
        return np.concatenate(y + dydt)


    def _get_bodies_initial_displacement(self) -> np.ndarray:
        """
        Calcule le vecteur d'état initial des corps libres (déplacements depuis référence).

        Returns:
            Array [Δu₁, Δθ₁, ..., Δuₙ, Δθₙ, v₁, ω₁, ..., vₙ, ωₙ]
        """
        y = []
        dydt = []
        for body in self.bodies:
            # Déplacements initiaux = position_init - position_ref
            y.append(body._initial_position - body._referencePosition)
            y.append(body._initial_angles - body._refAngles)
            # Vitesses initiales
            dydt.append(body._velocity)
            dydt.append(body._omega)
        return np.concatenate(y + dydt)



class MBSLinearSystem(__MBSBase):
    """
    Système multicorps en dynamique linéarisée (petits angles, petits déplacements).

    Hypothèses:
    - Angles < 10-15° (sin(θ) ≈ θ, cos(θ) ≈ 1)
    - Déplacements << dimensions caractéristiques
    - Matrices de transformation constantes

    Variables d'état:
    - U : déplacements [Δx, Δy, Δz, Δθx, Δθy, Δθz] depuis la configuration de référence
    - V : vitesses [vx, vy, vz, ωx, ωy, ωz]

    Configuration de référence:
    - Définie par les positions de référence des corps
    - Par définition : U_ref = 0 (tous les déplacements nuls)
    - Aucune précontrainte (F_ref = 0)

    Équation du mouvement:
    - M·dV/dt = -K·U - C·V + F_ext
    """

    def __init__(self):
        super().__init__()
        self._assembled = False

        # Matrices globales (à assembler par les classes dérivées)
        self._Kmatrix: Optional[np.ndarray] = None
        self._Cmatrix: Optional[np.ndarray] = None
        self._Mmatrix: Optional[np.ndarray] = None

        # Matrices de liaison (système de projection P·K·Q)
        self._Qmat_linkage: Optional[np.ndarray] = None  # Déplacements CDG → locaux
        self._Pmat_linkage: Optional[np.ndarray] = None  # Forces locales → CDG
        self._Kmat_linkage: Optional[np.ndarray] = None  # Raideur locale
        self._Cmat_linkage: Optional[np.ndarray] = None  # Amortissement local

        # Matrices partitionnées (libres/fixés)
        self._Mff: Optional[np.ndarray] = None
        self._invMff: Optional[np.ndarray] = None
        self._Kff: Optional[np.ndarray] = None
        self._Cff: Optional[np.ndarray] = None
        self._Kb: Optional[np.ndarray] = None
        self._Cb: Optional[np.ndarray] = None

        # Matrices de gap partitionnées
        self._Pgap_f: Optional[np.ndarray] = None
        self._Pgap_b: Optional[np.ndarray] = None
        self._Qgap_f: Optional[np.ndarray] = None
        self._Qgap_b: Optional[np.ndarray] = None

        # Jacobienne approximée
        self._Jac_linear: Optional[csc_matrix] = None

        # Gravité matricielle
        self._gravity_matrix: Optional[np.ndarray] = None

        # Positions de référence (pour reconstruction)
        self._yref: Optional[np.ndarray] = None
        self._yref_fixed: Optional[np.ndarray] = None

        self.__max_angle_threshold: Optional[float] = None

        self.__T_qr = None
        self.__qr_master_indices = None
        self.__free_unconstrained_indices = None

    def _block_slice(self, body_idx: int) -> slice:
        """Renvoie le slice pour le bloc 6×6 du corps i dans les matrices globales."""
        s = 6 * body_idx
        return slice(s, s + 6)

    def _index_bodies(self):
        """Crée le mapping corps → index dans les matrices globales."""
        self.body_index_map.clear()
        for i, b in enumerate(self._allbodies):
            self.body_index_map[b] = i

    def _vecProductMatrix(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Calcule la matrice antisymétrique du produit vectoriel [x×].

        Pour x = B - A, retourne la matrice [x×] tel que [x×]·v = x × v
        """
        x = B - A
        xi, yi, zi = x
        return np.array([
            [0, -zi, yi],
            [zi, 0, -xi],
            [-yi, xi, 0]
        ])

    def _assemble_linkage_matrices(self):
        """
        Assemble les matrices de transformation et de raideur des liaisons.

        Construit :
        - Q : projection déplacements CDG → déplacements locaux aux liaisons
        - K, C : raideur et amortissement locaux
        - P : transport forces locales → forces au CDG

        Relation : F_CDG = P @ K @ Q @ U_CDG
        """
        nbodies = len(self._allbodies)
        self._nlinks = len(self.links)

        # Matrices de liaison (projection P·K·Q)
        self._Qmat_linkage = np.zeros((6 * self._nlinks, 6 * nbodies))
        self._Pmat_linkage = np.zeros((6 * nbodies, 6 * self._nlinks))
        self._Kmat_linkage = np.zeros((6 * self._nlinks, 6 * self._nlinks))
        self._Cmat_linkage = np.zeros((6 * self._nlinks, 6 * self._nlinks))
        self._Kmat_kinematic = np.zeros((6 * self._nlinks, 6 * self._nlinks))

        # Cataloguer les liaisons par type
        self._non_linear_link = []
        self._linear_link = []
        self.__n_kinematic_links = 0

        # Liaisons avec gap (butées)
        ngap = len([link for link in self.links if link.HasGap])
        self._n_gapLink = ngap
        stop_delta_plus = np.ones(ngap * 6, dtype=float) * np.inf
        stop_delta_minus = -np.ones(ngap * 6, dtype=float) * np.inf
        Pgap = np.zeros((nbodies * 6, 6 * ngap))
        Qgap = np.zeros((ngap * 6, 6 * nbodies))
        Kgap = np.zeros((ngap * 6, ngap * 6))
        Cgap = np.zeros((ngap * 6, ngap * 6))
        id_gap = 0

        for id_link, link in enumerate(self.links):
            b1 = link.GetBody1
            b2 = link.GetBody2
            i = self.body_index_map[b1]
            j = self.body_index_map[b2]
            si = self._block_slice(i)
            sj = self._block_slice(j)

            # Positions de référence
            G1 = b1.GetReferencePosition()
            G2 = b2.GetReferencePosition()
            O1 = link.GetGlobalPoint1
            O2 = link.GetGlobalPoint2

            # Matrices de transformation (transport CDG ↔ point de liaison)
            A1 = np.eye(6)
            A2 = np.eye(6)

            # Vecteurs GO (du CDG vers le point de liaison)
            D_G1_O = self._vecProductMatrix(G1, O1)
            D_G2_O = self._vecProductMatrix(G2, O2)

            # Construction matrices A : [I, 0; [GO×], I]
            A1[3:, :3] = D_G1_O
            A2[3:, :3] = D_G2_O

            # Matrices B = A^T (pour la transformation inverse)
            B1 = A1.T
            B2 = A2.T

            # Matrices de raideur et amortissement locales
            Kt, Ct, Kth, Cth = link.GetLinearReactionMatrices

            Kloc = np.zeros((6, 6))
            Cloc = np.zeros((6, 6))
            Kloc[0:3, 0:3] = Kt
            Kloc[3:6, 3:6] = Kth
            Cloc[0:3, 0:3] = Ct
            Cloc[3:6, 3:6] = Cth

            # Cataloguer liaison linéaire/non-linéaire
            link_prop = (link, id_link, si, sj, A1, A2, B1, B2)
            if link.IsLinear:
                self._linear_link.append(link_prop)
            else:
                self._non_linear_link.append(link_prop)

            # Traitement des liaisons avec gap (butées)
            if link.HasGap:
                s_gap = slice(id_gap * 6, id_gap * 6 + 6)
                s_gap_trans = slice(id_gap * 6, id_gap * 6 + 3)
                s_gap_rot = slice(id_gap * 6 + 3, id_gap * 6 + 6)

                Qgap[s_gap, si] = -B1
                Qgap[s_gap, sj] = B2
                Pgap[si, s_gap] = -A1
                Pgap[sj, s_gap] = A2

                Kgap[s_gap, s_gap] += Kloc
                Cgap[s_gap, s_gap] += Cloc

                stop_delta_plus[s_gap_trans] = link.GetTransGap[:, 1]
                stop_delta_plus[s_gap_rot] = link.GetRotGap[:, 1]
                stop_delta_minus[s_gap_trans] = link.GetTransGap[:, 0]
                stop_delta_minus[s_gap_rot] = link.GetRotGap[:, 0]

                # Pour gap, on ne met pas dans les matrices globales
                Kloc = np.zeros((6, 6))
                Cloc = np.zeros((6, 6))
                id_gap += 1


            # Assemblage dans les matrices de liaison globales
            s_linkage = slice(id_link * 6, id_link * 6 + 6)

            # Q : U_local = Q @ U_CDG = -B1·U1 + B2·U2
            self._Qmat_linkage[s_linkage, si] += -B1
            self._Qmat_linkage[s_linkage, sj] += B2

            # P : F_CDG = P @ F_local = -A1·F_local sur corps1, +A2·F_local sur corps2
            self._Pmat_linkage[si, s_linkage] += -A1
            self._Pmat_linkage[sj, s_linkage] += A2

            # K, C locales
            self._Kmat_linkage[s_linkage, s_linkage] += Kloc
            self._Cmat_linkage[s_linkage, s_linkage] += Cloc

            if link.IsKinematic :
                self.__n_kinematic_links += 1
                self._Kmat_kinematic[s_linkage, s_linkage] += link.GetConstraintMatrix()
                # GetConstrainMatrix >> diagonal avec 1 si dll local bloqué


        # Matrices de gap (filtrer les gaps infinis)
        keepgap = (~np.isinf(stop_delta_plus)) & (~np.isinf(stop_delta_minus))
        self._Pmat_gap = Pgap[:, keepgap]
        self._Qmat_gap = Qgap[keepgap]
        self._Kmat_gap = Kgap[keepgap][:, keepgap]
        self._Cmat_gap = Cgap[keepgap][:, keepgap]
        self._gapPlus = stop_delta_plus[keepgap]
        self._gapMinus = stop_delta_minus[keepgap]

        # Matrices globales K et C
        self._Kmatrix = self._Pmat_linkage @ self._Kmat_linkage @ self._Qmat_linkage
        self._Cmatrix = self._Pmat_linkage @ self._Cmat_linkage @ self._Qmat_linkage

    def _assemble_mass_matrix(self):
        """Assemble la matrice de masse globale."""
        nbodies = len(self._allbodies)
        self._Mmatrix = np.zeros((6 * nbodies, 6 * nbodies), dtype=float)

        self._freedof = []
        self._fixeddof = []

        for k, body_k in enumerate(self._allbodies):
            s = self._block_slice(k)
            dof = list(range(s.start, s.start + 6))

            if not body_k.IsFixed:
                # Masse translation
                self._Mmatrix[s.start:s.start + 3, s.start:s.start + 3] = np.eye(3) * body_k._mass
                # Inertie rotation (au CDG)
                self._Mmatrix[s.start + 3:s.start + 6, s.start + 3:s.start + 6] = body_k._inertia
                self._freedof += dof
            else:
                self._fixeddof += dof

        # Gravité matricielle (répétée pour chaque corps libre)
        self._gravity_matrix = np.tile(
            np.concatenate([self.gravity, [0, 0, 0]]),
            self._nbodies
        )

    def _partition_matrices(self):
        """Partitionne les matrices en sous-systèmes libres/fixés."""
        if self._Kmatrix is None:
            return

        # Sous-matrices libres
        self._Mff = self._Mmatrix[self._freedof][:, self._freedof]
        self._invMff = np.linalg.inv(self._Mff)
        self._Kff = self._Kmatrix[self._freedof][:, self._freedof]
        self._Cff = self._Cmatrix[self._freedof][:, self._freedof]

        # Couplage avec corps fixés
        self._Kb = self._Kmatrix[self._freedof][:, self._fixeddof]
        self._Cb = self._Cmatrix[self._freedof][:, self._fixeddof]

        # Matrices de gap partitionnées
        self._Pgap_f = self._Pmat_gap[self._freedof]
        self._Pgap_b = self._Pmat_gap[self._fixeddof]
        self._Qgap_f = self._Qmat_gap[:, self._freedof]
        self._Qgap_b = self._Qmat_gap[:, self._fixeddof]



    def AssemblyMatrixSystem(self, print_report = False):
        """Assemble le système matriciel complet."""
        self._index_bodies()
        self._ntot = self._nbodies + self._nrefbodies
        self._assemble_mass_matrix()
        self._assemble_linkage_matrices()
        self._partition_matrices()
        self._assembled = True

        if print_report :
            print(f"\n{'=' * 60}")
            print("SYSTÈME ASSEMBLÉ")
            print(f"{'=' * 60}")
            print(f"Corps libres : {self._nbodies}")
            print(f"Corps fixes : {self._nrefbodies}")
            print(f"Liaisons : {len(self.links)}")
            print(f"DOF libres : {len(self._freedof)}")
            print(f"DOF fixés : {len(self._fixeddof)}")

    def CheckInitialTensions(self, t0: float = 0.0):
        """
        Vérifie les tensions initiales dans les liaisons.

        Affiche un avertissement si des forces/moments non nuls existent
        en configuration initiale.
        """
        if not self._assembled:
            raise ValueError("Système non assemblé")

        # États initiaux
        Dyfixed = self._get_fixedBodies_displacement_state(t0)
        Dy0 = self._get_bodies_initial_displacement()

        Ufixed = Dyfixed[:6 * self._nrefbodies]
        Vfixed = Dyfixed[6 * self._nrefbodies:]
        Uvec = Dy0[:6 * self._nbodies]
        Vvec = Dy0[6 * self._nbodies:]

        # Reconstruction état global
        U = np.zeros(6 * self._ntot, dtype=float)
        V = np.zeros_like(U, dtype=float)
        U[self._fixeddof] = Ufixed
        U[self._freedof] = Uvec
        V[self._fixeddof] = Vfixed
        V[self._freedof] = Vvec

        # Vérifier tensions dans chaque liaison
        init_tension = []
        for link, id_link, si, sj, A1, A2, B1, B2 in (self._non_linear_link + self._linear_link):
            Ui = U[si]
            Vi = V[si]
            Uj = U[sj]
            Vj = V[sj]

            Ui_point = B1 @ Ui
            Vi_point = B1 @ Vi
            Uj_point = B2 @ Uj
            Vj_point = B2 @ Vj

            if link.IsLinear:
                f = link.GetLinearLocalReactions(Ui_point, Vi_point, Uj_point, Vj_point)
            else:
                f = link.GetNonLinearLocalReactions(Ui_point, Vi_point, Uj_point, Vj_point)

            force, torque = f
            f_total = np.concatenate([force, torque])

            if np.any(np.abs(f_total) > 1e-6):
                b1 = link.GetBody1
                b2 = link.GetBody2
                init_tension.append((b1.GetName, b2.GetName, np.round(f_total, 5)))

        if len(init_tension) > 0:
            print("\n⚠️  Tensions initiales détectées dans les liaisons:")
            for b1, b2, f in init_tension:
                print(f"  {b1} >>> {b2} : [Force | Moment]")
                print(f"    {f}")






    def _get_bodies_reference_position(self) -> np.ndarray:
        """
        Retourne les positions de référence des corps libres.

        Returns:
            Array [x_ref₁, θ_ref₁, ..., x_refₙ, θ_refₙ, 0, 0, ..., 0]
        """
        y = []
        dydt = []
        for body in self.bodies:
            y.append(body._referencePosition)
            y.append(body._refAngles)
            dydt.append([0.] * 6)  # Vitesses nulles
        return np.concatenate(y + dydt)



    def _recompose_body_position(self, Dy: np.ndarray) -> np.ndarray:
        """
        Reconstruit les positions absolues depuis les déplacements.

        Args:
            Dy: Déplacements [Δu₁, Δθ₁, ..., v₁, ω₁, ...]

        Returns:
            Positions absolues [x₁, θ₁, ..., v₁, ω₁, ...]
        """
        y = Dy.copy()
        for i, body in enumerate(self.bodies):
            # Position = référence + déplacement
            y[6 * i:6 * i + 3] = body._referencePosition[:, None] + Dy[6 * i:6 * i + 3]
            y[6 * i + 3:6 * i + 6] = body._refAngles[:, None] + Dy[6 * i + 3:6 * i + 6]
        return y

    def _recompose_ref_body_position(self, Dy: np.ndarray) -> np.ndarray:
        """Reconstruit les positions absolues des corps fixes."""
        y = Dy.copy()
        for i, body in enumerate(self.ref_bodies):
            y[6 * i:6 * i + 3] = body._referencePosition[:, None] + Dy[6 * i:6 * i + 3]
            y[6 * i + 3:6 * i + 6] = body._refAngles[:, None] + Dy[6 * i + 3:6 * i + 6]
        return y

    def _check_angle_validity(self, Dy: np.ndarray) -> bool:
        """
        Vérifie que les angles restent dans le domaine de validité des petits angles.

        Args:
            Dy: Vecteur d'état [Δu, v]

        Returns:
            True si tous les angles sont < self.__max_angle_threshold
        """
        if self.__max_angle_threshold is None :
            return True
        max_angle_rad = self.__max_angle_threshold * np.pi / 180.0

        for i in range(self._nbodies):
            angles = Dy[6 * i + 3:6 * i + 6]
            if np.any(np.abs(angles) > max_angle_rad):
                print(f"\n⚠️  AVERTISSEMENT: Angles hors du domaine de validité!")
                print(f"   Corps {self.bodies[i].GetName}: θ = {np.rad2deg(angles)} °")
                print(f"   Limite de validité: ±{self.__max_angle_threshold}")
                print(f"   Les résultats peuvent être inexacts.")
                return False
        return True



    def _IVP_derivativeFunc(self, t: float, Dy: np.ndarray) -> np.ndarray:
        """
        Fonction dérivée pour l'intégrateur (dDy/dt = f(t, Dy)).

        Équation du mouvement (Option A - déplacements purs):
        M·dV/dt = -K·U - C·V + F_ext + F_nonlinear + F_gap

        Args:
            t: Temps
            Dy: Vecteur d'état [Δu₁, ..., Δuₙ, v₁, ..., vₙ]

        Returns:
            dDy/dt = [v₁, ..., vₙ, a₁, ..., aₙ]
        """
        # Extraction des déplacements et vitesses des corps fixes
        Dyfixed = self._get_fixedBodies_displacement_state(t)
        Ufixed = Dyfixed[:6 * self._nrefbodies]
        Vfixed = Dyfixed[6 * self._nrefbodies:]

        # Extraction des déplacements et vitesses des corps libres
        Uvec = Dy[:6 * self._nbodies]  # Déplacements [Δu, Δθ]
        Vvec = Dy[6 * self._nbodies:]  # Vitesses [v, ω]

        # Forces de réaction visco-élastiques LINÉAIRES
        # Option A: pas de F_ref car U_ref = 0 par définition
        # F = -K·U - C·V (signe moins car forces de rappel)
        F = -(self._Kff @ Uvec + self._Cff @ Vvec +
              self._Kb @ Ufixed + self._Cb @ Vfixed)


        # Ajout des forces NON-LINÉAIRES (frottement, etc.)
        if len(self._non_linear_link) > 0:
            # Déplacements locaux aux liaisons
            dUlocal = (self._Qmat_linkage[:, self._freedof] @ Uvec +
                       self._Qmat_linkage[:, self._fixeddof] @ Ufixed)
            dVlocal = (self._Qmat_linkage[:, self._freedof] @ Vvec +
                       self._Qmat_linkage[:, self._fixeddof] @ Vfixed)

            F += self._nonLinearForces(dUlocal, dVlocal)

        # Ajout des forces de contact GAP (butées)
        if self._n_gapLink > 0:
            F += self._penalizedGapContactReactionForces(Uvec, Vvec, Ufixed, Vfixed)

        # Accélérations = M⁻¹·F + g
        acc = self._invMff @ F + self._gravity_matrix


        # Retour: dDy/dt = [V, acc]
        return np.concatenate([Vvec, acc])

    def _nonLinearForces(self, duLocal: np.ndarray, dvLocal: np.ndarray) -> np.ndarray:
        """
        Calcule les forces non-linéaires (frottement, etc.).

        Args:
            duLocal: Déplacements locaux aux liaisons
            dvLocal: Vitesses locales aux liaisons

        Returns:
            Forces au CDG des corps libres
        """
        Flocal = np.zeros(self._nlinks * 6)

        for link, id_link, si, sj, A1, A2, B1, B2 in self._non_linear_link:
            s = slice(id_link * 6, id_link * 6 + 6)
            force, torque = link.GetNonLinearLocalReactions(
                dUlocal=duLocal[s],
                dVlocal=dvLocal[s]
            )
            Flocal[s] = np.concatenate([force, torque])

        # Transport au CDG
        F = self._Pmat_linkage @ Flocal
        return F[self._freedof]

    def _penalizedGapContactReactionForces(self, u: np.ndarray, v: np.ndarray,
                                           ub: np.ndarray, vb: np.ndarray) -> np.ndarray:
        """
        Calcule les forces de contact pénalisées (butées avec jeu).

        Args:
            u, v: Déplacements et vitesses corps libres
            ub, vb: Déplacements et vitesses corps fixes

        Returns:
            Forces de contact au CDG des corps libres
        """
        # Déplacements locaux aux gaps
        du = (self._Qgap_f @ u + self._Qgap_b @ ub)
        dv = (self._Qgap_f @ v + self._Qgap_b @ vb)

        # Pénétration (activation uniquement si gap violé)
        du_viol = (np.maximum(0., du - self._gapPlus) +
                   np.minimum(0., du - self._gapMinus))
        dv_viol = dv * (np.abs(du_viol) > 0.)

        # Forces de contact
        F = -self._Pgap_f @ (self._Kmat_gap @ du_viol + self._Cmat_gap @ dv_viol)

        return F

    def _approxJacobian(self, t: Optional[float] = None,
                        Dy: Optional[np.ndarray] = None) -> csc_matrix:
        """
        Calcule la jacobienne approximée pour l'intégrateur implicite.

        J = ∂f/∂Dy = [  0      I  ]
                     [-M⁻¹K  -M⁻¹C]

        Pour les gaps, ajoute une contribution dépendant de l'état.
        """
        if self._Jac_linear is None:
            n = 6 * self._nbodies
            A = np.zeros((2 * n, 2 * n))
            A[:n, n:] = np.eye(n)
            A[n:, :n] = -self._invMff @ self._Kff  # Signe + car F = -K·U
            A[n:, n:] = -self._invMff @ self._Cff  # Signe + car F = -C·V
            self._Jac_linear = csc_matrix(A)

        # Si pas de gap, jacobienne constante
        if self._n_gapLink == 0:
            return self._Jac_linear

        # Avec gaps, ajout de la contribution variable
        n = 6 * self._nbodies
        u = Dy[:n]

        Dyref = self._get_fixedBodies_displacement_state(t)
        ub = Dyref[:6 * self._nrefbodies]

        du = (self._Qgap_f @ u + self._Qgap_b @ ub)

        # Détection des gaps activés
        phi0 = np.maximum(0, du - self._gapPlus) + np.minimum(0, du - self._gapMinus)
        s_phi = 1.0 * (phi0 > 0) + 1.0 * (phi0 < 0)
        Pf = self._Pgap_f * s_phi[np.newaxis]

        # Contribution des gaps à la jacobienne
        Agap_penal = np.zeros((2 * n, 2 * n))
        Agap_penal[n:, :n] = -self._invMff @ (Pf @ (self._Kmat_gap @ self._Qgap_f))
        Agap_penal[n:, n:] = -self._invMff @ (Pf @ (self._Cmat_gap @ self._Qgap_f))
        Agap_penal = csc_matrix(Agap_penal)

        return Agap_penal + self._Jac_linear

    def RunDynamicSimulation(self, t_span: tuple, dt: float,
                             ode_method: str = "BDF",
                             print_step_rate: int = 0,
                             max_angle_threshold:(float|None) = None) -> tuple:
        """
        Lance la simulation dynamique du système.

        Args:
            t_span: Intervalle de temps (t_start, t_end)
            dt: Pas de temps
            ode_method: Méthode d'intégration ("BDF" ou "Radau")
            print_step_rate: Nombre de prints pendant simulation (0 = désactivé)

        Returns:
            (t_eval, results): Vecteur temps et dictionnaire des résultats par corps
        """
        self.__max_angle_threshold = max_angle_threshold

        if not self._assembled:
            self.AssemblyMatrixSystem()

        # Génération du vecteur temps
        nt = int(np.ceil((t_span[1] - t_span[0]) / dt)) + 1
        t_eval = np.linspace(t_span[0], t_span[1], nt)

        # État initial
        t0 = t_eval[0]
        Dy0 = self._get_bodies_initial_displacement()
        self._check_angle_validity(Dy0)

        # Jacobienne (constante si pas de gap)
        if self._n_gapLink == 0:
            jac = self._approxJacobian()
        else:
            jac = self._approxJacobian  # Fonction appelée à chaque pas

        # Configuration des prints
        if print_step_rate <= 1:
            substep = 1
            steps = np.array([0, nt], dtype=int)
        else:
            steps = np.unique([int(s) for s in np.linspace(0, nt, print_step_rate)])
            substep = len(steps) - 1

        # Allocation mémoire pour résultats
        Dy = np.zeros((self._nbodies * 12, nt))
        Dyfixed = np.zeros((self._nrefbodies * 12, nt))

        # Boucle d'intégration (par sous-intervalles si print activé)
        for k, (start_substep, end_substep) in enumerate(zip(steps[:-1], steps[1:]), start=1):
            t_substep = t_eval[start_substep:end_substep]
            t_span_sub = (t_substep[0], t_substep[-1])

            # Intégration
            sol = solve_ivp(
                self._IVP_derivativeFunc,
                t_span_sub,
                Dy0,
                method=ode_method,
                t_eval=t_substep,
                jac=jac,
            )

            # Stockage des résultats
            Dyfixed[:, start_substep:end_substep - 1] = np.array([
                self._get_fixedBodies_displacement_state(ti) for ti in t_substep[:-1]
            ]).T
            Dy[:, start_substep:end_substep - 1] = sol.y[:, :-1]

            # Préparation pas suivant
            Dy0 = sol.y[:, -1]

            # Dernière itération
            if k == substep:
                Dy[:, -1] = Dy0
                Dyfixed[:, -1] = self._get_fixedBodies_displacement_state(t_substep[-1])

            # Print progression
            if print_step_rate > 1:
                progress = k / substep * 100
                print(f"Simulation: {progress:.1f}% ({k}/{substep})")

        # Vérification finale de validité des angles
        self._check_angle_validity(Dy[:, -1])

        # Reconstruction des positions absolues
        y = self._recompose_body_position(Dy)
        yfixed = self._recompose_ref_body_position(Dyfixed)

        # Construction du dictionnaire de résultats
        result_dict = {}

        for body in self.ref_bodies:
            idx = self.ref_body_index[body.GetName]
            Dy_body = Dyfixed[6 * idx:6 * (idx + 1)]
            y_body = yfixed[6 * idx:6 * (idx + 1)]
            v_body = Dyfixed[6 * self._nrefbodies + 6 * idx:6 * self._nrefbodies + 6 * (idx + 1)]
            result_dict[body.GetName] = MBSBodySimulationResult(
                body, t_eval, Dy_body, y_body, v_body
            )

        for body in self.bodies:
            idx = self.body_index[body.GetName]
            Dy_body = Dy[6 * idx:6 * (idx + 1)]
            y_body = y[6 * idx:6 * (idx + 1)]
            v_body = Dy[6 * self._nbodies + 6 * idx:6 * self._nbodies + 6 * (idx + 1)]
            result_dict[body.GetName] = MBSBodySimulationResult(
                body, t_eval, Dy_body, y_body, v_body
            )

        return t_eval, result_dict



    def _check_linearity(self):

        if self._n_gapLink > 0 or len(self._non_linear_link) > 0:
            return False

        return True


    def __checkEigVals(self, lambda_, dll_vec = None):
        is_negative = (lambda_ < -1e-7)
        if np.any(is_negative):  # Tolérance pour erreurs numériques
            n_negative = [f"{x:.3e}" for x in lambda_[is_negative] ]
            if dll_vec is not None :
                axe = dll_vec[is_negative]
                n_negative = [f"{(a,n)}" for a,n in zip(axe, n_negative)]

            warnings.warn(f"Attention : \n{'\n'.join(n_negative)}\n"
                          "valeurs propres négatives détectées. "
                          f"Le système pourrait être instable ou mal conditionné.")

        # Clip les petites valeurs négatives dues aux erreurs numériques
        lambda_ = np.maximum(lambda_, 0.0)
        return lambda_

    def __all_dll(self, reference:bool = True):
        ref_dll = [f"{body.GetName} - {dof}"
             for body in self.ref_bodies
             for dof in ["x", "y", "z", "rx", "ry", "rz"]] if reference else []
        return np.array(
            ref_dll +\
            [f"{body.GetName} - {dof}"
             for body in self.bodies
             for dof in ["x", "y", "z", "rx", "ry", "rz"]]
        )


    def ComputeQrDecomposedSystem(self, print_infos=False,
                                  print_slaves_dof=False,
                                  rtol=None,
                                  drop_unconstrained_rows=True):
        """
        Décompose le système en identifiant les degrés de liberté maîtres et esclaves.

        Cette méthode utilise une décomposition QR avec pivotage pour détecter automatiquement
        les dépendances cinématiques dans le système (liaisons rigidifiées par pénalisation).
        Les DDL esclaves sont exprimés comme combinaisons linéaires des DDL maîtres, permettant
        une condensation du système qui élimine les modes parasites à haute fréquence.

        Contexte d'utilisation
        ----------------------
        Lorsqu'un système multi-corps contient des liaisons cinématiques rigidifiées par
        pénalisation (raideur k → ∞), la matrice de raideur K devient singulière ou très
        mal conditionnée. Cela génère :

        - Des valeurs propres négatives parasites (∼-1e-5)
        - Des fréquences propres irréalistes (>1e4 Hz)
        - Des instabilités numériques en simulation

        La décomposition QR identifie ces dépendances et construit une base réduite où
        seuls les DDL indépendants (maîtres) sont conservés.

        Principe mathématique
        ---------------------
        1. Extraction de K_cinematique (raideurs de pénalisation uniquement)
        2. Décomposition QR : K @ P = Q @ R avec pivotage
        3. Détection du rang numérique de R
        4. Construction de la relation x_slave = B @ x_master
        5. Assemblage de T_qr : x_freedof = T_qr @ x_master

        Classification des DDL
        ----------------------
        - **DDL totalement libres** : Aucune contrainte (K = 0, C = 0)
        - **DDL élastiques purs** : Contraints uniquement par ressorts (pas de contraintes cinématiques)
        - **DDL maîtres QR** : DDL indépendants dans les contraintes cinématiques
        - **DDL esclaves QR** : Déterminés algébriquement par les maîtres

        Parameters
        ----------
        print_infos : bool, optional
            Affiche un rapport détaillé de la décomposition incluant :
            - Nombre de DDL par catégorie
            - Rang numérique et conditionnement des matrices
            - Tests de validation de la décomposition
            - Avertissements si problèmes détectés
            Par défaut : False

        print_slaves_dof : bool, optional
            Affiche la liste complète des DDL esclaves identifiés.
            Utile pour comprendre quels DDL sont déterminés par d'autres.
            Par défaut : False

        rtol : float, optional
            Tolérance relative pour la détection du rang numérique.
            Si spécifiée : tolerance = rtol × σ_max
            Si None : tolerance automatique = σ_max × √ε × max(dimensions)
            où σ_max est la plus grande valeur singulière et ε la précision machine.
            Recommandé : laisser None (détection automatique).
            Par défaut : None

        drop_unconstrained_rows : bool, optional
            Si True, les DDL totalement libres (non contraints) sont automatiquement
            exclus de l'analyse et du système condensé.
            Si False, tous les DDL sont conservés (déconseillé).
            Par défaut : True

        """
        if not self._assembled:
            self.AssemblyMatrixSystem()

        if not self._check_linearity():
            raise ValueError("Le système n'est pas totalement linéaire. "
                             "Le système comporte des liaisons non linéaires ou "
                             "des liaisons de contact avec jeu. "
                             "L'analyse des degrés de liberté sur-contraints risque d'être faussée")

        if self.__n_kinematic_links == 0:
            raise ValueError("Le système n'a pas de liaisons cinématiques. "
                             "Pas de décomposition QR nécessaire")

        ndof_total = 6 * (self._nbodies + self._nrefbodies)
        all_indices = np.arange(ndof_total)
        dll_vec = self.__all_dll()

        # ===================================================================
        # ÉTAPE 1 : Identification des DLL totalement non contraints
        # ===================================================================
        if drop_unconstrained_rows:
            unconstrained_mask = np.isclose(self._Kmatrix, 0).all(axis=0)
        else:
            unconstrained_mask = np.full(ndof_total, False, dtype=bool)

        unconstrained_indices = all_indices[unconstrained_mask]

        # ===================================================================
        # ÉTAPE 2 : Matrice de raideur purement cinématique
        # ===================================================================
        K_cinematique = (self._Pmat_linkage @ self._Kmat_kinematic @ self._Qmat_linkage)
        # Augmentation artificielle des coefficients des DDL bloqués pour garantir
        # leur sélection comme DDL maîtres lors du pivotage QR (priorité aux référentiels)
        K_cinematique[self._fixeddof, :] *= 1e3
        K_cinematique[:, self._fixeddof] *= 1e3

        # ===================================================================
        # ÉTAPE 3 : Identification des DLL contraints uniquement par élasticité
        # ===================================================================
        elastic_only_mask = (~unconstrained_mask) & np.all(K_cinematique == 0, axis=0)
        elastic_only_indices = all_indices[elastic_only_mask]

        # ===================================================================
        # ÉTAPE 4 : DLL concernés par la décomposition QR
        # ===================================================================
        kinematic_constrained_mask = (~unconstrained_mask) & (~elastic_only_mask)
        kinematic_constrained_indices = all_indices[kinematic_constrained_mask]
        Kf_to_decompose = K_cinematique[kinematic_constrained_mask][:, kinematic_constrained_mask]
        ndof_qr = len(kinematic_constrained_indices)

        # ===================================================================
        # ÉTAPE 5 : Décomposition QR avec pivotage
        # ===================================================================
        Q, R, P = QR_decomposition(Kf_to_decompose, mode='economic', pivoting=True)

        # Détermination du rang avec SVD (plus robuste que diag(R))
        svd = np.linalg.svd(Kf_to_decompose, compute_uv=False)

        if rtol is not None:
            tolerance = rtol * svd.max()
        else:
            eps = np.finfo(float).eps
            tolerance = svd.max() * np.sqrt(eps) * max(Kf_to_decompose.shape)

        rank = np.sum(svd > tolerance)
        nslaves_qr = ndof_qr - rank

        # Conditionnement
        cond_K_cinematique = svd.max() / svd[svd > tolerance].min() if rank > 0 else np.inf

        if print_infos:
            print("=" * 70)
            print("DÉCOMPOSITION QR - IDENTIFICATION MAÎTRES/ESCLAVES")
            print("=" * 70)
            print(f"\n STRUCTURE DU SYSTÈME")
            print(f"  Degrés de liberté total               : {ndof_total}")
            print(f"  ├─ DDL totalement libres              : {len(unconstrained_indices)}")
            print(f"  ├─ DDL élastiques purs                : {len(elastic_only_indices)}")
            print(f"  └─ DDL avec contraintes cinématiques  : {ndof_qr}")
            print(f"      ├─ DDL maîtres QR                 : {rank}")
            print(f"      └─ DDL esclaves QR                : {nslaves_qr}")

            print(f"\n ANALYSE NUMÉRIQUE")
            print(f"  Conditionnement K_cinematique         : {cond_K_cinematique:.2e}")

            # Classification du conditionnement
            if cond_K_cinematique < 1e6:
                cond_status = " Excellent"
            elif cond_K_cinematique < 1e10:
                cond_status = " Acceptable"
            elif cond_K_cinematique < 1e15:
                cond_status = "️ Mal conditionné"
            else:
                cond_status = " Très mal conditionné"
            print(f"  Statut                                : {cond_status}")

            print(f"  Tolérance SVD appliquée               : {tolerance:.2e}")
            print(f"  Valeur singulière max                 : {svd.max():.2e}")
            print(f"  Valeur singulière min (>tol)          : {svd[svd > tolerance].min():.2e}")

            # Distribution des valeurs singulières
            print(f"\n  Distribution des valeurs singulières :")
            thresholds = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
            for thresh in thresholds:
                n_above = np.sum(svd > svd.max() * thresh)
                if n_above != ndof_qr:
                    print(f"    > {thresh:.0e} × σ_max : {n_above}/{ndof_qr}")

        # ===================================================================
        # ÉTAPE 6 : Construction de la matrice de couplage B
        # ===================================================================
        R_mm = R[:rank, :rank]
        R_ms = R[:rank, rank:]

        if nslaves_qr > 0:
            B, _, _, _ = np.linalg.lstsq(R_ms, -R_mm, rcond=None)
            # Méthode 1 : Pseudo-inverse (plus robuste)
            from scipy.linalg import pinv

            # On veut résoudre : R_ms @ B = -R_mm
            # Solution aux moindres carrés : B = pinv(R_ms) @ (-R_mm)
            # R_ms_pinv = pinv(R_ms)
            # B = R_ms_pinv @ (-R_mm)

            residual_matrix = R_ms @ B + R_mm
            residual_norm = np.linalg.norm(residual_matrix)
            residual_relative = residual_norm / np.linalg.norm(R_mm)

            if print_infos:
                print(f"\n CONSTRUCTION DE LA MATRICE DE COUPLAGE B")
                print(f"  Dimensions B                          : {B.shape}")
                print(f"  Résidu ||R_ms @ B + R_mm||            : {residual_norm:.2e}")
                print(f"  Résidu relatif                        : {residual_relative:.2e}")
                print(f"  Norme de B                            : {np.linalg.norm(B):.2e}")

                if residual_relative > 1e-6:
                    print(f"  Résidu élevé - vérifier le conditionnement")
                    print(f"    Cela indique probablement :")
                    print(f"      - Des liaisons cinématiques en série mal définies")
                    print(f"      - Des contraintes contradictoires")
                    print(f"      - Une boucle fermée avec dépendances")

        else:
            self.__T_qr = None
            self.__qr_master_indices = None
            self.__free_unconstrained_indices = None
            warnings.warn("Aucun DDL esclave détecté par QR. Système déjà de rang plein.")
            return

        # ===================================================================
        # ÉTAPE 7-9 : Construction de T_qr globale
        # ===================================================================
        T_c_local = np.zeros((ndof_qr, rank))
        T_c_local[:rank, :] = np.eye(rank)
        T_c_local[rank:, :] = B

        T_c_full = np.zeros((ndof_qr, rank))
        T_c_full[P, :] = T_c_local

        master_indices_qr_perm = np.arange(rank)
        master_indices_qr_orig = P[master_indices_qr_perm]
        qr_master_indices_global = kinematic_constrained_indices[master_indices_qr_orig]

        n_elastic = len(elastic_only_indices)
        nmaster_global = rank + n_elastic

        T_qr = np.zeros((ndof_total, nmaster_global))
        T_qr[np.ix_(kinematic_constrained_indices, np.arange(rank))] = T_c_full
        T_qr[np.ix_(elastic_only_indices, rank + np.arange(n_elastic))] = np.eye(n_elastic)

        # ===================================================================
        # ÉTAPE 10 : Stockage
        # ===================================================================
        self.__T_qr = T_qr
        master_indices_global = np.concatenate([elastic_only_indices, qr_master_indices_global])

        qr_freedof  = [i for i,m in enumerate(master_indices_global) if m in self._freedof]
        qr_fixeddof = [i for i,m in enumerate(master_indices_global) if m in self._fixeddof]
        master_freedof = [m for i, m in enumerate(master_indices_global) if m in self._freedof]
        master_fixeddof = [m for i, m in enumerate(master_indices_global) if m in self._fixeddof]
        free_unconstrained_indices = [i for i in unconstrained_indices if i in self._freedof]



        self.__qr_master_indices = master_indices_global
        self.__free_unconstrained_indices = free_unconstrained_indices
        self.__qr_freedof = qr_freedof
        self.__qr_fixeddof = qr_fixeddof
        self.__master_freedof = master_freedof
        self.__master_fixeddof = master_fixeddof



    def CheckUnconstrainedDegreeOfFreedom(self, print_infos=True):
        if not self._assembled :
            self.AssemblyMatrixSystem()

        if not self._check_linearity() :
            raise ValueError("Le système n'est pas totalement linéaire. "
                             "Le système comporte des liaisons non linéaires ou "
                             "des liaisons de contact avec jeu. "
                             "L'analyse des degrés de liberté non contraints risque d'être faussé")

        filtered_rows = np.all(self._Kff == 0, axis=0)
        self.__free_unconstrained_indices = np.array(list(range(6 * self._nbodies)))[filtered_rows]

        if print_infos :

            dll_body = ["X", "Y", "Z", "rX", "rY", "rZ"]
            system_dll = []
            for body in self.bodies :
                system_dll += [ body.GetName + " >>> " + dll for dll  in dll_body ]
            unconstrained_dll = np.array(system_dll)[filtered_rows]

            print("="*40)
            print("Dégrés de liberté libres et non-conditionnés : ")
            for s in unconstrained_dll :
                print(s)
            print("=" * 40)

    def ComputeNaturalPulsations(self, sort_values=False, drop_zeros=False, associated_dll=False):
        """
        Calcul des pulsations propres du système.

        Si une décomposition QR a été effectuée (ComputeQrDecomposedSystem),
        l'analyse est réalisée dans l'espace réduit des DDL maîtres,
        éliminant ainsi les fréquences parasites.

        Parameters
        ----------
        sort_values : bool, optional
            Active le tri croissant des valeurs. Par défaut : False
        drop_zeros : bool, optional
            Retire les lignes/colonnes totalement nulles de l'analyse. Par défaut : False
        associated_dll : bool, optional
            Retourne les DDL associés aux fréquences propres. Par défaut : False

        Returns
        -------
        omega : ndarray
            Pulsations propres [rad/s]
        dll_vec : ndarray, optional
            Vecteur des DDL associés (si associated_dll=True)

        See Also
        --------
        ComputeNaturalFrequencies : Conversion en fréquences [Hz]
        ComputeQrDecomposedSystem : Décomposition maître/esclave préalable
        """
        if not self._assembled:
            self.AssemblyMatrixSystem()

        if not self._check_linearity():
            raise ValueError("Le système n'est pas totalement linéaire. "
                             "Le système comporte des liaisons non linéaires ou "
                             "des liaisons de contact avec jeu. "
                             "L'analyse modale n'est pas possible.")

        # ===================================================================
        # BRANCHE 1 : Avec décomposition QR (espace réduit des maîtres)
        # ===================================================================
        use_qr = self.__T_qr is not None
        if use_qr :
            # Construction de la matrice de projection T_qr_ff
            # Lignes : maîtres free | Colonnes : dll free
            T_qr_ff = self.__T_qr[self._freedof][:, self.__qr_freedof]

            # Projection des matrices dans l'espace des maîtres free
            K = T_qr_ff.T @ self._Kff @ T_qr_ff
            M = T_qr_ff.T @ self._Mff @ T_qr_ff

            # Vecteur des DDL maîtres free
            dll_vec = self.__all_dll(reference=True)[self.__master_freedof]

        # ===================================================================
        # BRANCHE 2 : Sans décomposition QR (méthode classique)
        # ===================================================================
        else:
            dll_vec = self.__all_dll(reference=False)

            if drop_zeros:
                filtered_rows = ~np.all(self._Kff == 0, axis=0)
                K = self._Kff[filtered_rows][:, filtered_rows]
                M = self._Mff[filtered_rows][:, filtered_rows]
                dll_vec = dll_vec[filtered_rows]
            else:
                K = self._Kff
                M = self._Mff

        # ===================================================================
        # Résolution du problème aux valeurs propres généralisé
        # ===================================================================
        lambda_ = eigvalsh(K, M)
        lambda_ = self.__checkEigVals(lambda_, dll_vec)
        omega = np.sqrt(lambda_)

        if sort_values:
            args_sort = np.argsort(omega)
            omega = omega[args_sort]
            dll_vec = dll_vec[args_sort]

        if associated_dll:
            return omega, dll_vec
        return omega

    def ComputeModalAnalysis(self, sort_values = False,
                                   drop_zeros = False):
        """
        Réalise l'analyse modale du système.

        :param sort_values: (bool) active le tri croissant ou non des fréquences ;
        :param drop_zeros: (bool) retire les composantes nulles de l'analyse ;

        :return: modal_results(dict[body_name] = mode)
        """
        if not self._assembled:
            self.AssemblyMatrixSystem()

        if not self._check_linearity():
            raise ValueError("Le système n'est pas totalement linéaire. "
                             "Le système comporte des liaisons non linéaires ou "
                             "des liaisons de contact avec jeu. "
                             "L'analyse modale n'est pas possible.")

        filtered_rows = np.full(6 * self._nbodies, True, dtype=bool)
        # ===================================================================
        # BRANCHE 1 : Avec décomposition QR (espace réduit des maîtres)
        # ===================================================================
        use_qr = self.__T_qr is not None
        if use_qr:
            # Construction de la matrice de projection T_qr_ff
            # Lignes : maîtres free | Colonnes : dll free
            T_qr_ff = self.__T_qr[self._freedof][:, self.__qr_freedof]

            # Projection des matrices dans l'espace des maîtres free
            K = T_qr_ff.T @ self._Kff @ T_qr_ff
            M = T_qr_ff.T @ self._Mff @ T_qr_ff

            # Vecteur des DDL maîtres free
            dll_vec = self.__all_dll(reference=True)[self.__master_freedof]


        # ===================================================================
        # BRANCHE 2 : Sans décomposition QR (méthode classique)
        # ===================================================================
        else:
            dll_vec = self.__all_dll(reference=False)

            if drop_zeros:
                filtered_rows = ~np.all(self._Kff == 0, axis=0)
                K = self._Kff[filtered_rows][:, filtered_rows]
                M = self._Mff[filtered_rows][:, filtered_rows]
                dll_vec = dll_vec[filtered_rows]
            else:
                K = self._Kff
                M = self._Mff

        # ===================================================================
        # Résolution du problème aux valeurs propres généralisé
        # ===================================================================
        lambda_, phi_ = eigh(K, M)
        lambda_ = self.__checkEigVals(lambda_, dll_vec)

        n_mode = len(lambda_)

        # Normalisation  de phi_ pour obtenir
        # phi_.T @ Mf @ phi_ = Identité
        # phi_.T @ Kf @ phi_ = lambda_
        modal_mass = np.diag(phi_.T @ M @ phi_)
        phi_ = phi_ @ np.diag(1 / np.sqrt(modal_mass))

        # Frequencies and pulsations
        pulsation = np.sqrt(lambda_)

        if use_qr :
            phi_ = T_qr_ff @ phi_

        # Déplacements modaux
        modal_displacements = np.zeros((self._nbodies*6, n_mode))
        modal_displacements[filtered_rows] = phi_

        if sort_values :
            id_sort = np.argsort(pulsation)
            pulsation = pulsation[id_sort]
            modal_displacements = modal_displacements[:, id_sort]

        modal_results = MBSModalResults(pulsation,
                                        modal_displacements,
                                        [b.GetName for b in self.bodies])

        return modal_results




    def ComputeNaturalFrequencies(self,sort_values=False, drop_zeros=False, associated_dll = False):
        """
        Calcul des fréquences de propres du système.

        :param sort_values: (bool) active le tri croissant ou non des valeurs.
        :param drop_zeros: (bool) retire les colonnes et lignes totalement vides de l'analyse
        :param associated_dll: (bool) retourne les dll associés aux fréquences propres

        :return: numpy array des fréquences propres.

        See Also
        --------
        MBSLinearSystem.ComputeNaturalPulsations

        """
        r = self.ComputeNaturalPulsations(sort_values, drop_zeros, associated_dll)
        if associated_dll :
            return r[0] / (2 * np.pi), r[1]
        else :
            return r / (np.pi *2)

    def ComputeFrequencyDomainResponse(self, input_output: List,
                                       frequency_array: np.array = None,
                                       fstart=None, fend=None,
                                       print_damping=True,
                                       print_progress_step: int = None,
                                       nbase=20):
        if not self._assembled:
            self.AssemblyMatrixSystem()

        if not self._check_linearity():
            raise ValueError("Le système n'est pas totalement linéaire...")

        # ===================================================================
        # PARTIE 1 : Préparation des matrices
        # ===================================================================
        T_qr = self.__T_qr
        qr_decomposed = T_qr is not None

        if qr_decomposed:
            # ===============================================================
            # Construction de T_qr_ff : sous-matrice free → masters_free
            # ===============================================================

            # Extraction : lignes = freedof, colonnes = masters_free
            T_qr_ff = T_qr[np.ix_(self._freedof, self.__qr_freedof)]

            # Projection dans l'espace maîtres free
            K = T_qr_ff.T @ self._Kff @ T_qr_ff
            C = T_qr_ff.T @ self._Cff @ T_qr_ff
            M = T_qr_ff.T @ self._Mff @ T_qr_ff

            # Filtrage des colonnes de référence non nulles
            filtered_reference_cols = ~np.all((self._Kb == 0) & (self._Cb == 0), axis=0)

            # Projection des matrices de couplage
            Kb = T_qr_ff.T @ self._Kb[:, filtered_reference_cols]
            Cb = T_qr_ff.T @ self._Cb[:, filtered_reference_cols]

            # Vecteur des DDL (dans l'espace total avec référence)
            dll_vec_all = self.__all_dll(reference=True)
            dll_vec_reduced = dll_vec_all[self.__master_freedof]

        else:
            # ===============================================================
            # Sans QR : filtrer les DDL non contraints
            # ===============================================================
            filtered_rows = ~np.all((self._Kff == 0) & (self._Cff == 0), axis=0)
            filtered_reference_cols = ~np.all((self._Kb == 0) & (self._Cb == 0), axis=0)

            K = self._Kff[filtered_rows][:, filtered_rows]
            C = self._Cff[filtered_rows][:, filtered_rows]
            M = self._Mff[filtered_rows][:, filtered_rows]

            Kb = self._Kb[filtered_rows][:, filtered_reference_cols]
            Cb = self._Cb[filtered_rows][:, filtered_reference_cols]

            dll_vec = self.__all_dll(reference=False)
            dll_vec_reduced = dll_vec[filtered_rows]

            # Mapping pour reconstruction
            all_indexes = np.arange(self._nbodies * 6)
            selected_indexes = all_indexes[filtered_rows]

        # Mapping pour les colonnes de référence
        all_reference_indexes = np.arange(self._nrefbodies * 6)
        selected_reference_indexes = all_reference_indexes[filtered_reference_cols]
        reference_to_reduced = {idx_original: idx_reduced
                                for idx_reduced, idx_original
                                in enumerate(selected_reference_indexes)}

        # ===================================================================
        # PARTIE 2 : Analyse modale
        # ===================================================================
        lambda_, phi = eigh(K, M)
        lambda_ = self.__checkEigVals(lambda_, dll_vec_reduced)

        # Normalisation modale
        modal_mass = np.diag(phi.T @ M @ phi)
        phi = phi @ np.diag(1 / np.sqrt(modal_mass))

        # Pulsations propres
        omega_0 = np.sqrt(lambda_)

        # Projection dans la base modale
        Kb_phi = phi.T @ Kb
        Cb_phi = phi.T @ Cb
        Cphi = phi.T @ C @ phi

        # Détection amortissement diagonal
        non_diag_terms_norm = np.abs(Cphi - np.diag(np.diag(Cphi)))
        diag_terms_booleen = np.isclose(non_diag_terms_norm, 0.0)
        diag_damping = diag_terms_booleen.all()

        xi = None
        if diag_damping:
            xi = np.diag(Cphi) / (2 * omega_0)
            if print_damping:
                print("=" * 40)
                print("Damping factors")
                for wi, ci in zip(omega_0, xi):
                    print(f"ω0 = {wi:.4e} rad/s | ξ = {ci:.4e}")
        elif print_damping:
            non_zero_nondiag = non_diag_terms_norm[~diag_terms_booleen]
            print("=" * 40)
            print("Non diagonal damping coefficients :")
            print(f"Average : {non_zero_nondiag.mean():.2e}")
            print(f"Median : {np.median(non_zero_nondiag):.2e}")
            print(f"Max : {np.max(non_zero_nondiag):.2e}")

        # ===================================================================
        # PARTIE 3 : Validation et mapping des indices input_output
        # ===================================================================
        if not isinstance(input_output, list) or len(input_output) == 0:
            raise ValueError("'input_output' doit être une liste non vide de tuples "
                             "(corps_ref, axe_ref, corps_libre, axe_libre).")

        seen_pairs = set()
        index_rows_freedof = []  # Indices dans l'espace freedof ORIGINAL
        index_cols = []

        for item in input_output:
            # Validations de base
            if not isinstance(item, (tuple, list)) or len(item) != 4:
                raise ValueError(f"Chaque élément doit contenir 4 éléments "
                                 f"(corps1, axe1, corps2, axe2). Reçu : {item}")

            body1, axe1, body2, axe2 = item

            if not isinstance(body1, str) or not isinstance(body2, str):
                raise TypeError("Les noms de corps doivent être des chaînes.")

            if not isinstance(axe1, int) or not isinstance(axe2, int):
                raise TypeError("Les indices d'axes doivent être des entiers.")

            if not (0 <= axe1 <= 5) or not (0 <= axe2 <= 5):
                raise ValueError("Les indices d'axes doivent être entre 0 et 5.")

            if body1 not in self.ref_body_index:
                raise ValueError(f"'{body1}' n'est pas un corps de référence.")

            if body2 not in self.body_index:
                raise ValueError(f"'{body2}' n'est pas un corps libre.")

            # ===============================================================
            # Calcul des indices
            # ===============================================================
            id_ref = self.ref_body_index[body1]
            id_body = self.body_index[body2]

            # Indice dans l'espace de référence
            index_ref_original = id_ref * 6 + axe1

            # Indice dans l'espace des corps libres (freedof)
            index_body_original = id_body * 6 + axe2

            # ===============================================================
            # Vérifications de disponibilité
            # ===============================================================
            axe_names = ["x", "y", "z", "θx", "θy", "θz"]

            # Vérification corps de référence
            if index_ref_original not in selected_reference_indexes:
                raise ValueError(f"L'axe {axe_names[axe1]} du corps de référence '{body1}' "
                                 f"n'a pas de couplage d'excitation (K et C nuls).")

            # Vérification corps libre
            if qr_decomposed:
                # Interdire seulement les DDL totalement libres (unconstrained)
                if index_body_original in self.__free_unconstrained_indices:
                    raise ValueError(f"L'axe {axe_names[axe2]} du corps '{body2}' "
                                     f"est totalement libre (non contraint). "
                                     f"Pas de réponse fréquentielle définie.")
                # Les DDL esclaves sont OK, ils seront reconstruits via T_qr_ff
            else:
                # Sans QR : vérifier que le DDL n'est pas filtré
                if not filtered_rows[index_body_original]:
                    raise ValueError(f"L'axe {axe_names[axe2]} du corps '{body2}' "
                                     f"n'a pas de dynamique (K et C nuls).")

            # Vérification duplication
            pair_signature = (body1, axe1, body2, axe2)
            if pair_signature in seen_pairs:
                raise ValueError(f"Paire dupliquée détectée : {pair_signature}")
            seen_pairs.add(pair_signature)

            # Enregistrement des indices
            index_rows_freedof.append(index_body_original)
            index_cols.append(reference_to_reduced[index_ref_original])

        # ===================================================================
        # PARTIE 4 : Génération du vecteur de fréquences
        # ===================================================================
        if frequency_array is not None:
            if not np.all(frequency_array > 0):
                raise ValueError("'frequency_array' doit contenir uniquement des valeurs positives")
            npoints = len(frequency_array)
        elif (fstart is not None and fend is not None):
            if not fstart > 0:
                raise ValueError("'fstart' doit être positif")
            if not fend > fstart:
                raise ValueError("'fend' doit être supérieur à 'fstart'")
            npoints = int(nbase * fend / fstart)
            frequency_array = np.logspace(np.log10(fstart), np.log10(fend), npoints)
        else:
            # Plage automatique basée sur les fréquences propres
            omega_positive = omega_0[omega_0 > 0]
            if len(omega_positive) == 0:
                raise ValueError("Aucune fréquence propre positive détectée. "
                                 "Impossible de définir une plage automatique.")
            fstart = 0.5 * omega_positive.min() / (2 * np.pi)
            fend = 2.0 * omega_positive.max() / (2 * np.pi)
            npoints = int(nbase * fend / fstart)
            frequency_array = np.logspace(np.log10(fstart), np.log10(fend), npoints)

        # ===================================================================
        # PARTIE 5 : Calcul de la réponse fréquentielle
        # ===================================================================
        Gi_array = []
        print_progress_step = int(print_progress_step) if print_progress_step is not None else 0
        print_progress = print_progress_step > 0
        progress_step = [npoints * r / 100 for r in range(0, 100, print_progress_step)] if print_progress else []

        for i, wi in enumerate(frequency_array * 2 * np.pi):
            if len(progress_step) > 0 and i > progress_step[0]:
                print(f"     Progression >> {int(100 * i / npoints)} %")
                progress_step.pop(0)

            # Calcul dans l'espace modal
            if diag_damping:
                Hinv = np.diag(1 / (omega_0 ** 2 + 1j * wi * 2 * xi * omega_0 - wi ** 2))
            else:
                H = np.diag(omega_0 ** 2 - wi ** 2) + 1j * wi * Cphi
                Hinv = np.linalg.inv(H)

            # Réponse modale : shape (n_modes, n_excitations)
            Gqi = Hinv @ (Kb_phi + 1j * wi * Cb_phi)

            # Reconstruction dans l'espace de travail (masters_free ou filtered)
            # Shape: (n_workspace, n_excitations)
            Gi_workspace = phi @ Gqi

            # ===============================================================
            # Reconstruction dans l'espace freedof COMPLET
            # ===============================================================
            if qr_decomposed:
                # Reconstruire TOUS les DDL free (masters + esclaves) via T_qr_ff
                # T_qr_ff : (n_freedof, n_masters_free)
                # Gi_workspace : (n_masters_free, n_excitations)
                # → Gi_freedof : (n_freedof, n_excitations)
                Gi_freedof = T_qr_ff @ Gi_workspace
            else:
                # Sans QR : reconstruire avec zéros pour les DDL filtrés
                Gi_freedof = np.zeros((self._nbodies * 6, Gi_workspace.shape[1]),
                                      dtype=complex)
                Gi_freedof[selected_indexes, :] = Gi_workspace

            # Extraction des DDL demandés par l'utilisateur
            Gi_array.append(Gi_freedof[index_rows_freedof, index_cols])

        Gi_array = np.array(Gi_array)

        return MBSFrequencyDomainResult(frequency_array,
                                        Gi_array,
                                        input_output,
                                        omega_0,
                                        xi)

    def ___ComputeFrequencyDomainResponse(self, input_output: List,
                                       frequency_array: np.array = None,
                                       fstart=None, fend=None,
                                       print_damping=True,
                                       print_progress_step: int = None,
                                       nbase=20):
        if not self._assembled:
            self.AssemblyMatrixSystem()

        if not self._check_linearity():
            raise ValueError("Le système n'est pas totalement linéaire...")

        dll_vec = self.__all_dll(reference=False)

        # ===================================================================
        # PARTIE 1 : Préparation des matrices
        # ===================================================================
        T_qr = self.__T_qr
        qr_decomposed = T_qr is not None

        if qr_decomposed:
            # Travailler dans l'espace QR réduit
            K = T_qr.T @ self._Kff @ T_qr
            C = T_qr.T @ self._Cff @ T_qr
            M = T_qr.T @ self._Mff @ T_qr

            # Filtrage des colonnes nulles
            filtered_reference_cols = ~np.all((self._Kb == 0) & (self._Cb == 0), axis=0)

            # Projection dans l'espace QR
            Kb = T_qr.T @ self._Kb[:, filtered_reference_cols]
            Cb = T_qr.T @ self._Cb[:, filtered_reference_cols]

            dll_vec_reduced = dll_vec[self.__qr_master_indices]

        else:
            # Sans QR : filtrer les DDL non contraints
            filtered_rows = ~np.all((self._Kff == 0) & (self._Cff == 0), axis=0)
            filtered_reference_cols = ~np.all((self._Kb == 0) & (self._Cb == 0), axis=0)

            K = self._Kff[filtered_rows][:, filtered_rows]
            C = self._Cff[filtered_rows][:, filtered_rows]
            M = self._Mff[filtered_rows][:, filtered_rows]

            Kb = self._Kb[filtered_rows][:, filtered_reference_cols]
            Cb = self._Cb[filtered_rows][:, filtered_reference_cols]

            dll_vec_reduced = dll_vec[filtered_rows]

            # Mapping pour reconstruction
            all_indexes = np.arange(self._nbodies * 6)
            selected_indexes = all_indexes[filtered_rows]

        # Mapping pour les colonnes de référence
        all_reference_indexes = np.arange(self._nrefbodies * 6)
        selected_reference_indexes = all_reference_indexes[filtered_reference_cols]
        reference_to_reduced = {idx_original: idx_reduced
                                for idx_reduced, idx_original
                                in enumerate(selected_reference_indexes)}

        # ===================================================================
        # PARTIE 2 : Analyse modale
        # ===================================================================
        lambda_, phi = eigh(K, M)
        lambda_ = self.__checkEigVals(lambda_, dll_vec_reduced)

        # Normalisation modale
        modal_mass = np.diag(phi.T @ M @ phi)
        phi = phi @ np.diag(1 / np.sqrt(modal_mass))

        # Pulsations propres
        omega_0 = np.sqrt(lambda_)

        # Projection dans la base modale
        Kb_phi = phi.T @ Kb
        Cb_phi = phi.T @ Cb
        Cphi = phi.T @ C @ phi

        # Détection amortissement diagonal
        non_diag_terms_norm = np.abs(Cphi - np.diag(np.diag(Cphi)))
        diag_terms_booleen = np.isclose(non_diag_terms_norm, 0.0, atol=0., rtol=1e-3)
        diag_damping = diag_terms_booleen.all()

        xi = None
        if diag_damping:
            xi = np.diag(Cphi) / (2 * omega_0)
            if print_damping:
                print("=" * 40)
                print("Damping factors")
                for wi, ci in zip(omega_0, xi):
                    print(f"ω0 = {wi:.4e} rad/s | ξ = {ci:.4e}")
        elif print_damping:
            non_zero_nondiag = non_diag_terms_norm[~diag_terms_booleen]
            print("=" * 40)
            print("Non diagonal damping coefficients :")
            print(f"Average : {non_zero_nondiag.mean():.2e}")
            print(f"Median : {np.median(non_zero_nondiag):.2e}")
            print(f"Max : {np.max(non_zero_nondiag):.2e}")

        # ===================================================================
        # PARTIE 3 : Validation et mapping des indices input_output
        # ===================================================================
        if not isinstance(input_output, list) or len(input_output) == 0:
            raise ValueError("'input_output' doit être une liste non vide...")

        seen_pairs = set()
        index_rows_freedof = []  # Indices dans l'espace freedof ORIGINAL
        index_cols = []

        for item in input_output:
            # Validations de base
            if not isinstance(item, (tuple, list)) or len(item) != 4:
                raise ValueError(f"Chaque élément doit contenir 4 éléments. Reçu : {item}")

            body1, axe1, body2, axe2 = item

            if not isinstance(body1, str) or not isinstance(body2, str):
                raise TypeError("Les noms de corps doivent être des chaînes.")

            if not isinstance(axe1, int) or not isinstance(axe2, int):
                raise TypeError("Les indices d'axes doivent être des entiers.")

            if not (0 <= axe1 <= 5) or not (0 <= axe2 <= 5):
                raise ValueError("Les indices d'axes doivent être entre 0 et 5.")

            if body1 not in self.ref_body_index:
                raise ValueError(f"'{body1}' n'est pas un corps de référence.")

            if body2 not in self.body_index:
                raise ValueError(f"'{body2}' n'est pas un corps libre.")

            # Calcul des indices dans l'espace freedof ORIGINAL
            id_ref = self.ref_body_index[body1]
            id_body = self.body_index[body2]

            index_ref_original = id_ref * 6 + axe1
            index_body_original = id_body * 6 + axe2

            # Vérifications de disponibilité
            if index_ref_original not in selected_reference_indexes:
                axe_names = ["x", "y", "z", "θx", "θy", "θz"]
                raise ValueError(f"L'axe {axe_names[axe1]} du corps '{body1}' "
                                 f"n'a pas de couplage d'excitation.")

            # CORRECTION : Vérifier seulement que ce n'est pas un DDL totalement libre
            if qr_decomposed:
                # Interdire seulement les DDL totalement libres (unconstrained)
                if index_body_original in self.__free_unconstrained_indices:
                    axe_names = ["x", "y", "z", "θx", "θy", "θz"]
                    raise ValueError(f"L'axe {axe_names[axe2]} du corps '{body2}' "
                                     f"est totalement libre (non contraint). "
                                     f"Pas de réponse fréquentielle définie.")
                # Les esclaves sont OK, ils seront reconstruits par T_qr
            else:
                # Sans QR : vérifier dans filtered_rows
                if not filtered_rows[index_body_original]:
                    axe_names = ["x", "y", "z", "θx", "θy", "θz"]
                    raise ValueError(f"L'axe {axe_names[axe2]} du corps '{body2}' "
                                     f"n'a pas de dynamique.")

            # Vérification duplication
            pair_signature = (body1, axe1, body2, axe2)
            if pair_signature in seen_pairs:
                raise ValueError(f"Paire dupliquée : {pair_signature}")
            seen_pairs.add(pair_signature)

            # Enregistrer les indices dans l'espace freedof ORIGINAL
            index_rows_freedof.append(index_body_original)
            index_cols.append(reference_to_reduced[index_ref_original])

        # ===================================================================
        # PARTIE 4 : Génération du vecteur de fréquences
        # ===================================================================
        if frequency_array is not None:
            if not np.all(frequency_array > 0):
                raise ValueError("'frequency_array' must be positive")
            npoints = len(frequency_array)
        elif (fstart is not None and fend is not None):
            if not fstart > 0:
                raise ValueError("'fstart' must be positive")
            if not fend > fstart:
                raise ValueError("'fend' must be greater than 'fstart'")
            npoints = int(nbase * fend / fstart)
            frequency_array = np.logspace(np.log10(fstart), np.log10(fend), npoints)
        else:
            fstart = 0.5 * omega_0[omega_0 > 0].min() / (2 * np.pi)
            fend = 2.0 * omega_0[omega_0 > 0].max() / (2 * np.pi)
            npoints = int(nbase * fend / fstart)
            frequency_array = np.logspace(np.log10(fstart), np.log10(fend), npoints)

        # ===================================================================
        # PARTIE 5 : Calcul de la réponse fréquentielle
        # ===================================================================
        Gi_array = []
        print_progress_step = int(print_progress_step) if print_progress_step is not None else 0
        print_progress = print_progress_step > 0
        progress_step = [npoints * r / 100 for r in range(0, 100, print_progress_step)] if print_progress else []

        for i, wi in enumerate(frequency_array * 2 * np.pi):
            if len(progress_step) > 0 and i > progress_step[0]:
                print(f"     Progression >> {int(100 * i / npoints)} %")
                progress_step.pop(0)

            # Calcul dans l'espace modal
            if diag_damping:
                Hinv = np.diag(1 / (omega_0 ** 2 + 1j * wi * 2 * xi * omega_0 - wi ** 2))
            else:
                H = np.diag(omega_0 ** 2 - wi ** 2) + 1j * wi * Cphi
                Hinv = np.linalg.inv(H)

            # Réponse modale
            Gqi = Hinv @ (Kb_phi + 1j * wi * Cb_phi)

            # Reconstruction dans l'espace de travail (QR réduit ou filtré)
            Gi_workspace = phi @ Gqi  # Shape: (n_workspace, n_excitations)

            # CORRECTION CRITIQUE : Reconstruction dans l'espace freedof complet
            if qr_decomposed:
                # Reconstruire TOUS les DDL (masters + esclaves) avec T_qr
                Gi_freedof = T_qr @ Gi_workspace  # Shape: (n_freedof, n_excitations)
            else:
                # Sans QR : reconstruire avec zéros pour les DDL filtrés
                Gi_freedof = np.zeros((self._nbodies * 6, Gi_workspace.shape[1]), dtype=complex)
                Gi_freedof[selected_indexes, :] = Gi_workspace

            # Extraction des indices demandés dans l'espace freedof complet
            Gi_array.append(Gi_freedof[index_rows_freedof, index_cols])

        Gi_array = np.array(Gi_array)

        return MBSFrequencyDomainResult(frequency_array,
                                        Gi_array,
                                        input_output,
                                        omega_0,
                                        xi)

