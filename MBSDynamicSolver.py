import numpy as np
from typing import Optional
from scipy.sparse import csc_matrix
from scipy.integrate import solve_ivp

from MultiBodySimulation.MBSSimulationResults import MBSBodySimulationResult

class MBSDynamicSolver:
    """
    Classe de base pour les solveurs dynamiques de systèmes multicorps.

    Responsabilités:
    - Calcul des dérivées temporelles
    - Calcul des forces (linéaires, non-linéaires, gap)
    - Gestion de la jacobienne
    - Intégration temporelle

    Les classes dérivées implémentent différentes stratégies de résolution.
    """

    def __init__(self, system: 'MBSLinearSystem'):
        """
        Parameters
        ----------
        system : MBSLinearSystem
            Système multicorps assemblé
        """
        if not system._assembled:
            raise ValueError("Le système doit être assemblé avant création du solveur")

        self.system = system
        self._max_angle_threshold: Optional[float] = None

    def compute_forces(self, t: float, Uvec: np.ndarray, Vvec: np.ndarray,
                       Ufixed: np.ndarray, Vfixed: np.ndarray) -> np.ndarray:
        """
        Calcule les forces totales appliquées aux corps libres.

        Parameters
        ----------
        t : float
            Temps actuel
        Uvec : ndarray
            Déplacements corps libres [Δu, Δθ]
        Vvec : ndarray
            Vitesses corps libres [v, ω]
        Ufixed : ndarray
            Déplacements corps fixes
        Vfixed : ndarray
            Vitesses corps fixes

        Returns
        -------
        F : ndarray
            Forces totales au CDG des corps libres
        """
        # Forces visco-élastiques linéaires
        F = -(self.system._Kff @ Uvec + self.system._Cff @ Vvec +
              self.system._Kb @ Ufixed + self.system._Cb @ Vfixed)

        # Forces non-linéaires
        if len(self.system._non_linear_link) > 0:
            dUlocal = (self.system._Qmat_linkage[:, self.system._freedof] @ Uvec +
                       self.system._Qmat_linkage[:, self.system._fixeddof] @ Ufixed)
            dVlocal = (self.system._Qmat_linkage[:, self.system._freedof] @ Vvec +
                       self.system._Qmat_linkage[:, self.system._fixeddof] @ Vfixed)
            F += self._compute_nonlinear_forces(dUlocal, dVlocal)

        # Forces de contact gap
        if self.system._n_gapLink > 0:
            F += self._compute_gap_forces(Uvec, Vvec, Ufixed, Vfixed)

        return F

    def _compute_nonlinear_forces(self, duLocal: np.ndarray,
                                  dvLocal: np.ndarray) -> np.ndarray:
        """Calcule les forces non-linéaires (frottement, etc.)"""
        Flocal = np.zeros(self.system._nlinks * 6)

        for link, id_link, si, sj, A1, A2, B1, B2 in self.system._non_linear_link:
            s = slice(id_link * 6, id_link * 6 + 6)
            force, torque = link.GetNonLinearLocalReactions(
                dUlocal=duLocal[s],
                dVlocal=dvLocal[s]
            )
            Flocal[s] = np.concatenate([force, torque])

        F = self.system._Pmat_linkage @ Flocal
        return F[self.system._freedof]

    def _compute_gap_forces(self, u: np.ndarray, v: np.ndarray,
                            ub: np.ndarray, vb: np.ndarray) -> np.ndarray:
        """Calcule les forces de contact pénalisées (butées avec jeu)"""
        du = (self.system._Qgap_f @ u + self.system._Qgap_b @ ub)
        dv = (self.system._Qgap_f @ v + self.system._Qgap_b @ vb)

        du_viol = (np.maximum(0., du - self.system._gapPlus) +
                   np.minimum(0., du - self.system._gapMinus))
        dv_viol = dv * (np.abs(du_viol) > 0.)

        F = -self.system._Pgap_f @ (self.system._Kmat_gap @ du_viol +
                                    self.system._Cmat_gap @ dv_viol)
        return F

    def compute_derivative(self, t: float, Dy: np.ndarray) -> np.ndarray:
        """
        Calcule la dérivée temporelle dDy/dt = f(t, Dy).

        Parameters
        ----------
        t : float
            Temps actuel
        Dy : ndarray
            Vecteur d'état [Δu₁, ..., Δuₙ, v₁, ..., vₙ]

        Returns
        -------
        dDy_dt : ndarray
            Dérivée [v₁, ..., vₙ, a₁, ..., aₙ]
        """
        # États des corps fixes
        Dyfixed = self.system._get_fixedBodies_displacement_state(t)
        Ufixed = Dyfixed[:6 * self.system._nrefbodies]
        Vfixed = Dyfixed[6 * self.system._nrefbodies:]

        # États des corps libres
        n = 6 * self.system._nbodies
        Uvec = Dy[:n]
        Vvec = Dy[n:]

        # Forces totales
        F = self.compute_forces(t, Uvec, Vvec, Ufixed, Vfixed)

        # Accélérations
        acc = self.system._invMff @ F + self.system._gravity_matrix

        return np.concatenate([Vvec, acc])

    def compute_jacobian(self, t: Optional[float] = None,
                         Dy: Optional[np.ndarray] = None) -> csc_matrix:
        """
        Calcule la jacobienne ∂f/∂Dy pour l'intégrateur implicite.

        Returns
        -------
        J : csc_matrix
            Jacobienne du système
        """
        raise NotImplementedError("À implémenter dans les classes dérivées")

    def solve(self, t_span: tuple, dt: float, **kwargs) -> tuple:
        """
        Résout le système dynamique.

        Parameters
        ----------
        t_span : tuple
            Intervalle (t_start, t_end)
        dt : float
            Pas de temps
        **kwargs : dict
            Options spécifiques au solveur

        Returns
        -------
        t_eval : ndarray
            Vecteur temps
        results : dict
            Résultats par corps
        """
        raise NotImplementedError("À implémenter dans les classes dérivées")

    def _check_angle_validity(self, Dy: np.ndarray) -> bool:
        """Vérifie la validité des petits angles"""
        if self._max_angle_threshold is None:
            return True

        max_angle_rad = self._max_angle_threshold * np.pi / 180.0

        for i in range(self.system._nbodies):
            angles = Dy[6 * i + 3:6 * i + 6]
            if np.any(np.abs(angles) > max_angle_rad):
                body = self.system.bodies[i]
                print(f"\n⚠️  AVERTISSEMENT: Angles hors validité!")
                print(f"   Corps {body.GetName}: θ = {np.rad2deg(angles)}°")
                print(f"   Limite: ±{self._max_angle_threshold}°")
                return False
        return True


class MBSScipyIVPSolver(MBSDynamicSolver):
    """
    Solveur utilisant scipy.integrate.solve_ivp (méthode actuelle).

    Supporte les méthodes BDF et Radau avec jacobienne approximée.
    """

    def compute_jacobian(self, t: Optional[float] = None,
                         Dy: Optional[np.ndarray] = None) -> csc_matrix:
        """
        Jacobienne approximée J = [0, I; -M⁻¹K, -M⁻¹C].

        Ajoute la contribution des gaps si présents et état fourni.
        """
        # Jacobienne linéaire de base (calculée une seule fois)
        if not hasattr(self, '_Jac_linear') or self._Jac_linear is None:
            n = 6 * self.system._nbodies
            A = np.zeros((2 * n, 2 * n))
            A[:n, n:] = np.eye(n)
            A[n:, :n] = -self.system._invMff @ self.system._Kff
            A[n:, n:] = -self.system._invMff @ self.system._Cff
            self._Jac_linear = csc_matrix(A)

        # Sans gap, jacobienne constante
        if self.system._n_gapLink == 0:
            return self._Jac_linear

        # Avec gaps, contribution variable
        if t is None or Dy is None:
            return self._Jac_linear

        n = 6 * self.system._nbodies
        u = Dy[:n]

        Dyref = self.system._get_fixedBodies_displacement_state(t)
        ub = Dyref[:6 * self.system._nrefbodies]

        du = (self.system._Qgap_f @ u + self.system._Qgap_b @ ub)

        # Détection gaps activés
        phi0 = (np.maximum(0, du - self.system._gapPlus) +
                np.minimum(0, du - self.system._gapMinus))
        s_phi = 1.0 * (phi0 > 0) + 1.0 * (phi0 < 0)
        Pf = self.system._Pgap_f * s_phi[np.newaxis]

        # Contribution gap à la jacobienne
        Agap = np.zeros((2 * n, 2 * n))
        Agap[n:, :n] = -self.system._invMff @ (Pf @ (self.system._Kmat_gap @ self.system._Qgap_f))
        Agap[n:, n:] = -self.system._invMff @ (Pf @ (self.system._Cmat_gap @ self.system._Qgap_f))

        return csc_matrix(Agap) + self._Jac_linear

    def solve(self, t_span: tuple, dt: float,
              method: str = "BDF",
              print_step_rate: int = 0,
              max_angle_threshold: Optional[float] = None) -> tuple:
        """
        Intégration avec scipy.solve_ivp.

        Parameters
        ----------
        t_span : tuple
            Intervalle (t_start, t_end)
        dt : float
            Pas de temps
        method : str
            Méthode d'intégration ("BDF" ou "Radau")
        print_step_rate : int
            Nombre de prints de progression (0 = désactivé)
        max_angle_threshold : float, optional
            Seuil de validité des angles [degrés]

        Returns
        -------
        t_eval : ndarray
            Vecteur temps
        results : dict
            Résultats par corps {nom: MBSBodySimulationResult}
        """
        self._max_angle_threshold = max_angle_threshold

        # Génération vecteur temps
        nt = int(np.ceil((t_span[1] - t_span[0]) / dt)) + 1
        t_eval = np.linspace(t_span[0], t_span[1], nt)

        # État initial
        t0 = t_eval[0]
        Dy0 = self.system._get_bodies_initial_displacement()
        self._check_angle_validity(Dy0)

        # Jacobienne (constante si pas de gap)
        if self.system._n_gapLink == 0:
            jac = self.compute_jacobian()
        else:
            jac = self.compute_jacobian  # Fonction appelée à chaque pas

        # Configuration prints
        if print_step_rate <= 1:
            substep = 1
            steps = np.array([0, nt], dtype=int)
        else:
            steps = np.unique([int(s) for s in np.linspace(0, nt, print_step_rate)])
            substep = len(steps) - 1

        # Allocation résultats
        Dy = np.zeros((self.system._nbodies * 12, nt))
        Dyfixed = np.zeros((self.system._nrefbodies * 12, nt))

        # Boucle d'intégration
        for k, (start_substep, end_substep) in enumerate(zip(steps[:-1], steps[1:]), start=1):
            t_substep = t_eval[start_substep:end_substep]
            t_span_sub = (t_substep[0], t_substep[-1])

            # Intégration
            sol = solve_ivp(
                self.compute_derivative,
                t_span_sub,
                Dy0,
                method=method,
                t_eval=t_substep,
                jac=jac,
            )

            # Stockage
            Dyfixed[:, start_substep:end_substep - 1] = np.array([
                self.system._get_fixedBodies_displacement_state(ti)
                for ti in t_substep[:-1]
            ]).T
            Dy[:, start_substep:end_substep - 1] = sol.y[:, :-1]

            # Préparation pas suivant
            Dy0 = sol.y[:, -1]

            # Dernière itération
            if k == substep:
                Dy[:, -1] = Dy0
                Dyfixed[:, -1] = self.system._get_fixedBodies_displacement_state(t_substep[-1])

            # Progression
            if print_step_rate > 1:
                progress = k / substep * 100
                print(f"Simulation: {progress:.1f}% ({k}/{substep})")

        # Vérification finale
        self._check_angle_validity(Dy[:, -1])

        # Reconstruction positions absolues
        y = self.system._recompose_body_position(Dy)
        yfixed = self.system._recompose_ref_body_position(Dyfixed)

        # Construction résultats
        results = self._build_results_dict(t_eval, Dy, y, Dyfixed, yfixed)

        return t_eval, results

    def _build_results_dict(self, t_eval: np.ndarray,
                            Dy: np.ndarray, y: np.ndarray,
                            Dyfixed: np.ndarray, yfixed: np.ndarray) -> dict:
        """Construit le dictionnaire de résultats par corps"""
        results = {}

        # Corps fixes
        for body in self.system.ref_bodies:
            idx = self.system.ref_body_index[body.GetName]
            Dy_body = Dyfixed[6 * idx:6 * (idx + 1)]
            y_body = yfixed[6 * idx:6 * (idx + 1)]
            v_body = Dyfixed[6 * self.system._nrefbodies + 6 * idx:
                             6 * self.system._nrefbodies + 6 * (idx + 1)]
            results[body.GetName] = MBSBodySimulationResult(
                body, t_eval, Dy_body, y_body, v_body
            )

        # Corps libres
        for body in self.system.bodies:
            idx = self.system.body_index[body.GetName]
            Dy_body = Dy[6 * idx:6 * (idx + 1)]
            y_body = y[6 * idx:6 * (idx + 1)]
            v_body = Dy[6 * self.system._nbodies + 6 * idx:
                        6 * self.system._nbodies + 6 * (idx + 1)]
            results[body.GetName] = MBSBodySimulationResult(
                body, t_eval, Dy_body, y_body, v_body
            )

        return results


class MBSQRReducedSolver(MBSDynamicSolver):
    """
    Solveur utilisant la décomposition QR pour réduire le système.

    Résout uniquement les DDL maîtres, reconstruisant ensuite les esclaves.

    À IMPLÉMENTER : structure prête pour intégration future.
    """

    def __init__(self, system: 'MBSLinearSystem'):
        super().__init__(system)

        # Vérifications
        if system.__dict__.get('_MBSLinearSystem__T_qr') is None:
            raise ValueError(
                "La décomposition QR doit être effectuée avant "
                "(appeler system.ComputeQrDecomposedSystem())"
            )

        # Récupération matrices réduites
        self.T_qr = system._MBSLinearSystem__T_qr
        self.qr_freedof = system._MBSLinearSystem__qr_freedof
        self.qr_fixeddof = system._MBSLinearSystem__qr_fixeddof

        # TODO: Pré-calculer les matrices réduites
        # self._compute_reduced_matrices()

    def solve(self, t_span: tuple, dt: float, **kwargs) -> tuple:
        """À IMPLÉMENTER"""
        raise NotImplementedError(
            "Solveur QR en cours de développement. "
            "Utilisez MBSScipyIVPSolver pour le moment."
        )


class MBSConstraintStabilizedSolver(MBSDynamicSolver):
    """
    Solveur BDF2 avec stabilisation de contraintes (Baumgarte, projection).

    Vérifie et corrige la non-violation des contraintes cinématiques.

    À IMPLÉMENTER : structure prête pour intégration future.
    """

    def __init__(self, system: 'MBSLinearSystem',
                 stabilization_method: str = "baumgarte"):
        """
        Parameters
        ----------
        stabilization_method : str
            Méthode de stabilisation ("baumgarte", "projection", "lagrange")
        """
        super().__init__(system)
        self.stabilization_method = stabilization_method

        # TODO: Extraire les contraintes cinématiques
        # self._extract_kinematic_constraints()

    def solve(self, t_span: tuple, dt: float, **kwargs) -> tuple:
        """À IMPLÉMENTER"""
        raise NotImplementedError(
            "Solveur avec stabilisation de contraintes en cours de développement. "
            "Utilisez MBSScipyIVPSolver pour le moment."
        )