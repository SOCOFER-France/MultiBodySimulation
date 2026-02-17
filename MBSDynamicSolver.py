import numpy as np
from typing import Optional,Tuple

import pypardiso
from scipy.sparse import csc_matrix, eye, bmat
from scipy.integrate import solve_ivp
from scipy.linalg import solve as lin_solver

pypardiso_installed = False
try :
    from pypardiso import spsolve

    pypardiso_installed = True
except :
    print("Intsall PyPardiso")


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

        sol = solve_ivp(
            self.compute_derivative,
            t_span,
            Dy0,
            method=method,
            t_eval=t_eval,
            jac=jac,
        )

        # Allocation résultats
        Dy = sol.y
        Dyfixed = np.array([
                self.system._get_fixedBodies_displacement_state(ti)
                for ti in t_eval
            ]).T


        # Vérification finale
        self._check_angle_validity(Dy[:, -1])

        # Reconstruction positions absolues
        y = self.system._recompose_body_position(Dy)
        yfixed = self.system._recompose_ref_body_position(Dyfixed)

        # Construction résultats
        results = self._build_results_dict(t_eval, Dy, y, Dyfixed, yfixed)

        return t_eval, results




def linearSolver(matA,vecB):
    """
    Résout un système linéaire avec le solveur direct PyPardiso. si installé
    Sinon utilise scipy

    Paramètres
    ----------
    matA : ndarray ou matrice creuse
        Matrice du système.
    vecB : ndarray
        Vecteur du second membre.
    x0 : ndarray, optionnel
        Solution initiale (par défaut None).
    precond : bool, optionnel
        Non utilisé ici (présent pour compatibilité).
    rtol : float, optionnel
        Non utilisé ici (présent pour compatibilité).
    atol : float, optionnel
        Non utilisé ici (présent pour compatibilité).

    Retourne
    -------
    ndarray
        Solution du système.
    """
    if pypardiso_installed:
        return spsolve(matA,vecB)
    else :
        return lin_solver(matA, vecB)


def compute_bdf2_coefficients(tn, tn_minus_1, tn_minus_2):
    """
    Calcule les coefficients de BDF2 à pas variable.

    La discrétisation BDF2 transforme dX/dt = F(X,Y,t) en :
        Xn = alpha * F(Xn, Yn, tn) + beta

    où :
        beta = beta_nm1 * X_{n-1} + beta_nm2 * X_{n-2}

    Paramètres
    ----------
    tn : float
        Temps au pas actuel n
    tn_minus_1 : float
        Temps au pas n-1
    tn_minus_2 : float
        Temps au pas n-2

    Retourne
    --------
    alpha : float
        Coefficient multiplicatif de F(Xn, Yn, tn)
    beta_nm1 : float
        Coefficient de X_{n-1}
    beta_nm2 : float
        Coefficient de X_{n-2}

    Notes
    -----
    Les coefficients sont calculés selon :
        h_n = tn - tn_minus_1          (pas actuel)
        h_{n-1} = tn_minus_1 - tn_minus_2  (pas précédent)
        ρ = h_n / h_{n-1}              (ratio des pas)

    Formules BDF2 :
        alpha = h_n * (1 + ρ) / (1 + 2ρ)
        beta_nm1 = (1 + ρ)² / (1 + 2ρ)
        beta_nm2 = -ρ² / (1 + 2ρ)

    Propriété : beta_nm1 + beta_nm2 = 1 (conservation pour état stationnaire)

    Exemples
    --------
    >>> # Pas constant h = 0.1
    >>> alpha, b1, b2 = compute_bdf2_coefficients(1.0, 0.9, 0.8)
    >>> print(f"alpha={alpha:.4f}, beta_nm1={b1:.4f}, beta_nm2={b2:.4f}")
    alpha=0.0667, beta_nm1=1.3333, beta_nm2=-0.3333

    >>> # Pas variable (h_n=0.1, h_{n-1}=0.05)
    >>> alpha, b1, b2 = compute_bdf2_coefficients(1.0, 0.9, 0.85)
    >>> print(f"rho={0.1/0.05}, sum(betas)={b1+b2:.6f}")
    rho=2.0, sum(betas)=1.000000
    """

    if tn_minus_2 is None :
        # Euler implicite
        return tn - tn_minus_1, 1.0, 0.

    # Validation des arguments
    if tn <= tn_minus_1:
        raise ValueError(f"tn ({tn}) doit être > tn_minus_1 ({tn_minus_1})")
    if tn_minus_1 <= tn_minus_2:
        raise ValueError(f"tn_minus_1 ({tn_minus_1}) doit être > tn_minus_2 ({tn_minus_2})")

    # Calcul des pas de temps
    h_n = tn - tn_minus_1  # Pas actuel
    h_n_minus_1 = tn_minus_1 - tn_minus_2  # Pas précédent

    # Ratio des pas
    rho = h_n / h_n_minus_1

    # Dénominateur commun
    denom = 1.0 + 2.0 * rho

    # Coefficients BDF2
    alpha = h_n * (1.0 + rho) / denom
    beta_nm1 = (1.0 + rho) ** 2 / denom
    beta_nm2 = -(rho ** 2) / denom

    return alpha, beta_nm1, beta_nm2


def pi_step_controller(h_old, err_n, err_nm1,
                       k_I=0.4, k_P=0.3, safety=0.9):
    """
    Contrôleur PI pour le pas de temps.

    Formule : h_new = h_old * safety * (target/err_n)^k_I * (err_n/err_{n-1})^k_P

    Paramètres :
    - k_I : gain intégral (0.3-0.5)
    - k_P : gain proportionnel (0.2-0.4)
    - safety : facteur de sécurité (0.8-0.95)
    """
    if err_n < 1e-12:
        return h_old * 2.0  # Erreur nulle → doubler

    # Composante intégrale
    factor_I = (1.0 / err_n) ** k_I

    # Composante proportionnelle (si disponible)
    if err_nm1 is not None and err_nm1 > 1e-12:
        factor_P = (err_nm1 / err_n) ** k_P
    else:
        factor_P = 1.0

    # Nouveau pas
    h_new = h_old * safety * factor_I * factor_P

    # Limites de variation
    h_new = np.clip(h_new, h_old * 0.2, h_old * 5.0)

    return h_new

class MBSConstraintStabilizedSolver(MBSDynamicSolver):
    """
    Solveur BDF2 avec stabilisation de contraintes cinématiques.

    Méthodes disponibles:
    - None : pas de stabilisation
    """

    def __init__(self, system: 'MBSLinearSystem',
                 constraint_tolerance: float = 1e-8,
                 atol: float = 1e-5,
                 tol: float =1e-5,
                 maxInnerIter: int = 100,
                 inner_atol = 1e-8,
                 inner_tol = 1e-8,):

        super().__init__(system)

        self.constraint_tolerance = float(constraint_tolerance)
        self.inner_atol = float(inner_atol)
        self.inner_tol = float(inner_tol)
        self.atol = float(atol)
        self.tol = float(tol)
        self.maxIter = int(maxInnerIter)

        n = 6 * self.system._nbodies
        A = np.zeros((2 * n, 2 * n))
        A[:n, n:] = np.eye(n)
        A[n:, :n] = -self.system._invMff @ self.system._Kff
        A[n:, n:] = -self.system._invMff @ self.system._Cff
        self._Jac_linear = csc_matrix(A)

        self.bF_linear = np.zeros((2 * n, n))
        self.bF_linear[n:,] = self.system._invMff

        Kkin_f = self.system._Kkin_f
        self.ncons = Kkin_f.shape[0]
        if self.ncons > 0 :
            Kkin_b = self.system._Kkin_b

            Jkin_f = np.zeros((self.ncons, 2 * n))
            Jkin_f[:, :n] = Kkin_f

            nref = self.system._nrefbodies * 6
            Jkin_b = np.zeros((self.ncons, 2 * nref))

            Jkin_b[:, :nref] = Kkin_b

            self.Jkin_f = csc_matrix(Jkin_f)
            self.Jkin_b = Jkin_b

    def polynomialExtrapolator(self, tnp1, tprev, xprev):

        if len(tprev) < 3:
            return xprev[0]

        t = tprev[:3]
        x = xprev[:3]

        # Interpolation de Lagrange à l'ordre 2 (3 points)
        def L(k):
            idx = [i for i in range(3) if i != k]
            return ((tnp1 - t[idx[0]]) * (tnp1 - t[idx[1]])) / ((t[k] - t[idx[0]]) * (t[k] - t[idx[1]]))

        xnp1 = sum(x[k] * L(k) for k in range(3))
        return xnp1

    def solve(self, t_span: tuple, dt: float,
              max_angle_threshold: Optional[float] = None,
              adaptative = False,
              stabilization_method: (str,None) = "Lagrangian",
              ) -> tuple:
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

        stabilization_method = "" if stabilization_method is None else stabilization_method

        self._max_angle_threshold = max_angle_threshold

        # Génération vecteur temps
        nt = int(np.ceil((t_span[1] - t_span[0]) / dt)) + 1

        # État initial
        tn = t_span[0]
        Dy_n = self.system._get_bodies_initial_displacement()
        tm = None
        Dy_m = np.zeros_like(Dy_n, dtype=float)


        self._check_angle_validity(Dy_n)

        # Vecteurs solution
        n = self.system._nbodies * 6 * 2
        nref = self.system._nrefbodies * 6 * 2

        n_estimate = max(100, nt * 3)

        t_eval = np.zeros(n_estimate, dtype=float)
        Dy = np.zeros((n, n_estimate), dtype=float)
        Dyfixed = np.zeros((nref, n_estimate), dtype=float)

        Dy[:, 0] = Dy_n
        Dyfixed[:, 0] = self.system._get_fixedBodies_displacement_state(tn)
        t_eval[0] = tn

        err = 0.

        dhmin = dt * 0.1

        if adaptative:
            dh = dhmin * 1.1
        else :
            dh = dt

        if self.ncons > 0:
            Flambda_n = np.zeros(self.ncons, dtype = float)

        i = 0
        while t_eval[i] < t_span[1]:

            # Capacité dépassée ? Doubler la taille
            if i + 1 >= n_estimate:
                Dy = np.hstack([Dy, np.zeros_like(Dy)])
                Dyfixed = np.hstack([Dyfixed, np.zeros_like(Dyfixed)])
                t_eval = np.hstack([t_eval, np.zeros_like(t_eval)])
                n_estimate *= 2

            tnp1 = min(tn + dh, t_span[1])

            if adaptative:
                Dy_predict = self._RK2_step(tnp1, tn, Dy_n, Dyfixed[:,i])
            else :
                Dy_predict = Dy_n[:]

            if self.ncons > 0 and stabilization_method == "Lagrangian" :
                Dynp1, Dyfixed_np1, Flambda_n = self._BDF2_Lagrangian(tnp1,
                                                                      tn,
                                                                      tm,
                                                                      Dy_predict,
                                                                      Dy_n,
                                                                      Dy_m,
                                                                      Flambda_n)

            else :
                Dynp1, Dyfixed_np1 = self._BDF2_step(tnp1,
                                                 tn,
                                                 tm,
                                                 Dy_n,
                                                 Dy_n,
                                                 Dy_m)

            err_old = err
            err = np.linalg.norm(Dynp1 - Dy_predict) / (self.atol + self.tol * np.linalg.norm(Dynp1))

            t_eval[i+1] = tnp1
            Dy[:,i+1] = Dynp1
            Dyfixed[:,i+1] = Dyfixed_np1

            Dy_m = Dy_n[:]
            Dy_n = Dynp1[:]
            tm = tn
            tn = tnp1

            i += 1
            if adaptative:
                dhmin = dt * 0.1
                dh = pi_step_controller(dh, err, err_old)
                dh = np.clip(dh, dhmin, dt)


        # Vérification finale
        self._check_angle_validity(Dy[:, -1])

        Dy = Dy[:,:i+1]
        Dyfixed = Dyfixed[:,:i+1]
        t_eval = t_eval[:i+1]

        # Reconstruction positions absolues
        y = self.system._recompose_body_position(Dy)
        yfixed = self.system._recompose_ref_body_position(Dyfixed)

        # Construction résultats
        results = self._build_results_dict(t_eval, Dy, y, Dyfixed, yfixed)

        return t_eval, results


    def compute_forces(self, t: float, Uvec: np.ndarray, Vvec: np.ndarray,
                       Ufixed: np.ndarray, Vfixed: np.ndarray) -> np.ndarray:
        """
        Calcule les forces non-liénaire + gap appliquées aux corps libres.

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
        F = self.system._Mff @ self.system._gravity_matrix - self.system._Kb @ Ufixed - self.system._Cb @ Vfixed

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


    def _RK2_step(self,tnp1: float, tn: float,
                  Dy_n: np.ndarray,Dyfixed_n : np.ndarray):

        # États des corps fixes
        nref = 6* self.system._nrefbodies
        Un_fixed = Dyfixed_n[:nref]
        Vn_fixed = Dyfixed_n[nref:]

        # États des corps libres
        n = 6 * self.system._nbodies
        Un = Dy_n[:n]
        Vn = Dy_n[n:]

        dh = tnp1 - tn

        Fn = self.compute_forces(tn, Un, Vn, Un_fixed, Vn_fixed)
        Dyn_tilda = Dy_n + dh/2 *( self._Jac_linear @ Dy_n + self.bF_linear @ Fn )

        Dyfixed_tilda = self.system._get_fixedBodies_displacement_state(tn + dh/2)
        Ftilda = self.compute_forces(tn, Dyn_tilda[:n], Dyn_tilda[n:], Dyfixed_tilda[:nref], Dyfixed_tilda[nref:])

        Dynp1 = Dy_n + dh *( self._Jac_linear @ Dyn_tilda + self.bF_linear @ Ftilda )

        return Dynp1

    def _BDF2_step(self,tnp1: float, tn: float, tm: float,
                   Dy: np.ndarray, Dy_n: np.ndarray, Dy_m: np.ndarray) :
        # États des corps fixes
        Dyfixed = self.system._get_fixedBodies_displacement_state(tnp1)
        Ufixed = Dyfixed[:6 * self.system._nrefbodies]
        Vfixed = Dyfixed[6 * self.system._nrefbodies:]

        # États des corps libres
        n = 6 * self.system._nbodies
        Uvec = Dy[:n]
        Vvec = Dy[n:]


        alpha, beta_n, beta_m =  compute_bdf2_coefficients(tnp1, tn, tm)
        #x_next = alpha * f(..) + beta
        Bvec = Dy_n * beta_n + Dy_m * beta_m
        Fnp1 = self.compute_forces(tnp1, Uvec, Vvec, Ufixed, Vfixed)
        Amat = eye(2 * n, format = "csc") - alpha * self._Jac_linear

        b = alpha * self.bF_linear @ Fnp1 + Bvec
        r =  Amat @ Dy - b

        scale = np.maximum( np.abs(Dy), 1.0)
        tol = self.inner_tol * np.linalg.norm(Dy / scale)
        res = np.linalg.norm(r / scale) / (self.inner_atol + tol)

        iter = 0
        while res > 1.0 and iter < self.maxIter  :
            iter += 1

            Dy = linearSolver(Amat, b)

            Uvec = Dy[:n]
            Vvec = Dy[n:]
            Fnp1 = self.compute_forces(tnp1, Uvec, Vvec, Ufixed, Vfixed)
            b = alpha * self.bF_linear @ Fnp1 + Bvec
            r = Amat @ Dy - b
            res = np.linalg.norm(r / scale) / (self.inner_atol + tol)

        print(f"time = {tnp1}, iter = {iter}, res = {res}")
        return Dy, Dyfixed

    def _BDF2_Lagrangian(self, tnp1: float, tn: float, tm: float,
                         Dy: np.ndarray, Dy_n: np.ndarray, Dy_m: np.ndarray,
                         Flambda_n) :
        # États des corps fixes
        Dyfixed = self.system._get_fixedBodies_displacement_state(tnp1)
        nref = 6 * self.system._nrefbodies
        Ufixed = Dyfixed[:nref]
        Vfixed = Dyfixed[nref:]

        # États des corps libres
        n = 6 * self.system._nbodies
        Uvec = Dy[:n]
        Vvec = Dy[n:]


        alpha, beta_n, beta_m =  compute_bdf2_coefficients(tnp1, tn, tm)
        #x_next = alpha * f(..) + beta
        Bvec = Dy_n * beta_n + Dy_m * beta_m
        Fnp1 = self.compute_forces(tnp1, Uvec, Vvec, Ufixed, Vfixed)
        Amat_lin = eye(2 * n, format = "csc") - alpha * self._Jac_linear

        Amat = bmat([[Amat_lin, alpha * self.Jkin_f.T],
                           [self.Jkin_f, None]],
                    format = "csc")

        bDyn = alpha * self.bF_linear @ Fnp1 + Bvec
        bCons = - self.Jkin_b @ Dyfixed
        b = np.hstack([bDyn, bCons])

        Dz = np.hstack([Dy, Flambda_n])
        Dz_prev = Dz[:]

        r = Amat @ Dz - b

        scale = np.maximum( np.abs(Dz), 1.0)
        tol = self.inner_tol * np.linalg.norm(Dz / scale)
        res = np.linalg.norm(r / scale) / (self.inner_atol + tol)


        iter = 0
        while res > 1.0 and iter < self.maxIter  :
            iter += 1

            Dz = linearSolver(Amat, b)

            Uvec = Dz[:n]
            Vvec = Dz[n:2*n]
            Fnp1 = self.compute_forces(tnp1, Uvec, Vvec, Ufixed, Vfixed)
            bDyn = alpha * self.bF_linear @ Fnp1 + Bvec
            b[:2*n] = bDyn
            r = Amat @ Dz - b

            res = np.linalg.norm(r / scale) / (self.inner_atol + tol)
            dx = np.linalg.norm( (Dz-Dz_prev) / scale )
            if dx < self.inner_atol + tol :
                break
            Dz_prev = Dz[:]

        Dy = Dz[:2*n]
        Flambda_n = Dz[2*n:]



        return Dy, Dyfixed, Flambda_n


