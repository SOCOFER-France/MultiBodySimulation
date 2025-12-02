import numpy as np
from typing import Dict, List, Optional
from scipy.sparse import csc_matrix
from scipy.integrate import solve_ivp
from scipy.linalg import eigvalsh, eigh
import warnings

from MultiBodySimulation.MBSMechanicalJoint import _MBSLink3D
from MultiBodySimulation.MBSBody import MBSRigidBody3D, MBSReferenceBody3D
from MultiBodySimulation.MBSSimulationResults import MBSBodySimulationResult, MBSModalResults



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

        # Cataloguer les liaisons par type
        self._non_linear_link = []
        self._linear_link = []

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


    def __checkEigVals(self, lambda_):
        if np.any(lambda_ < -1e-10):  # Tolérance pour erreurs numériques
            n_negative = np.sum(lambda_ < -1e-10)
            warnings.warn(f"Attention : {n_negative} valeurs propres négatives détectées. "
                          f"Le système pourrait être instable ou mal conditionné.")

        # Clip les petites valeurs négatives dues aux erreurs numériques
        lambda_ = np.maximum(lambda_, 0.0)
        return lambda_


    def ComputeModalAnalysis(self, sort_values = False,
                                   drop_zeros = False):
        """
        Réalise l'analyse modale du système.

        :param sort_values: (bool) active le tri croissant ou non des fréquences ;
        :param drop_zeros: (bool) retire les composantes nulles de l'analyse ;

        :return: modal_results(dict[body_name] = mode)
        """
        if not self._assembled :
            self.AssemblyMatrixSystem()

        if not self._check_linearity() :
            raise ValueError("Le système n'est pas totalement linéaire. "
                             "Le système comporte des liaisons non linéaires ou "
                             "des liaisons de contact avec jeu. "
                             "L'analyse modale n'est pas possible.")

        if drop_zeros :
            filtered_rows = ~ np.all( self._Kff == 0, axis=0)
            K = self._Kff[filtered_rows][:,filtered_rows]
            M = self._Mff[filtered_rows][:,filtered_rows]
        else :
            filtered_rows = np.full(self._nbodies*6,True, dtype=bool)
            K = self._Kff
            M = self._Mff

        lambda_, phi_ = eigh(K, M)
        lambda_ = self.__checkEigVals(lambda_)

        n_mode = len(lambda_)

        # Normalisation  de phi_ pour obtenir
        # phi_.T @ Mf @ phi_ = Identité
        # phi_.T @ Kf @ phi_ = lambda_
        modal_mass = np.diag(phi_.T @ M @ phi_)
        phi_ = phi_ @ np.diag(1 / np.sqrt(modal_mass))

        # Frequencies and pulsations
        pulsation = np.sqrt(lambda_)

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


    def ComputeNaturalPulsations(self,sort_values=False, drop_zeros=False):
        """
        Calcul des pulsations de propres du système.

        :param sort_values: (bool) active le tri croissant ou non des valeurs.
        :param drop_zeros: (bool) retire les colonnes et lignes totalement vides de l'analyse

        :return: numpy array des pulsations propres.

        See Also
        --------
        MBSLinearSystem.ComputeNaturalFrequencies

        """
        if not self._assembled :
            self.AssemblyMatrixSystem()

        if not self._check_linearity() :
            raise ValueError("Le système n'est pas totalement linéaire. "
                             "Le système comporte des liaisons non linéaires ou "
                             "des liaisons de contact avec jeu. "
                             "L'analyse modale n'est pas possible.")

        if drop_zeros :
            filtered_rows = ~ np.all( self._Kff == 0, axis=0)
            K = self._Kff[filtered_rows][:,filtered_rows]
            M = self._Mff[filtered_rows][:,filtered_rows]
        else :
            K = self._Kff
            M = self._Mff

        lambda_ = eigvalsh(K,M)
        lambda_ = self.__checkEigVals(lambda_)
        omega = np.sqrt(lambda_)

        if sort_values :
            return np.sort(omega)
        return omega

    def ComputeNaturalFrequencies(self,sort_values=False, drop_zeros=False):
        """
        Calcul des fréquences de propres du système.

        :param sort_values: (bool) active le tri croissant ou non des valeurs.
        :param drop_zeros: (bool) retire les colonnes et lignes totalement vides de l'analyse

        :return: numpy array des fréquences propres.

        See Also
        --------
        MBSLinearSystem.ComputeNaturalPulsations

        """
        return self.ComputeNaturalPulsations(sort_values, drop_zeros)/(2*np.pi)



