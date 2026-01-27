import numpy as np
from scipy.linalg import qr as QR_decomposition
import warnings

def RotationMatrix(theta_x, theta_y, theta_z):
    """
    Args:
        theta_x (float): Angle de roulis (rotation autour de l'axe X).
        theta_y (float): Angle de tangage (rotation autour de l'axe Y).
        theta_z (float): Angle de lacet (rotation autour de l'axe Z).

    Returns:
        numpy.ndarray: Matrice de rotation 3x3.
    """
    # Matrice de rotation autour de l'axe X (roll)
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])

    # Matrice de rotation autour de l'axe Y (pitch)
    R_y = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])

    # Matrice de rotation autour de l'axe Z (yaw)
    R_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])

    # Combinaison des rotations : R = R_z * R_y * R_x
    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def _identify_protected_dof(fixeddof, kinematic_constrained_indices):
    """
    Identifie les DDL fixés qui sont contraints cinématiquement et doivent être protégés.

    Parameters
    ----------
    fixeddof : array-like
        Indices globaux des DDL fixés (excitations du système)
    kinematic_constrained_indices : array-like
        Indices globaux des DDL contraints cinématiquement

    Returns
    -------
    protected_global : ndarray
        Indices globaux des DDL à protéger
    protected_local : ndarray
        Indices locaux (dans kinematic_constrained_indices) des DDL à protéger
    """
    # Intersection : DDL fixés ET cinématiquement contraints
    protected_global = np.array([i for i in fixeddof if i in kinematic_constrained_indices])

    # Mapping global → local dans kinematic_constrained_indices
    kinematic_dict = {val: idx for idx, val in enumerate(kinematic_constrained_indices)}
    protected_local = np.array([kinematic_dict[i] for i in protected_global])

    return protected_global, protected_local


def _create_reordered_matrix(K_cinematique, kinematic_constrained_indices, protected_local):
    """
    Réorganise la matrice cinématique pour placer les DDL protégés en premier.

    Parameters
    ----------
    K_cinematique : ndarray
        Matrice de raideur cinématique complète (ndof_total × ndof_total)
    kinematic_constrained_indices : ndarray
        Indices globaux des DDL contraints cinématiquement
    protected_local : ndarray
        Indices locaux des DDL à protéger

    Returns
    -------
    K_reordered : ndarray
        Matrice réorganisée (ndof_qr × ndof_qr)
    reorder_map : ndarray
        Permutation appliquée : [protected_local, free_to_slave_local]
    n_protected : int
        Nombre de DDL protégés
    """
    ndof_qr = len(kinematic_constrained_indices)

    # Séparer protected et free_to_slave
    all_local_indices = np.arange(ndof_qr)
    free_to_slave_local = np.array([i for i in all_local_indices if i not in protected_local])

    # Créer la permutation : [protected en tête | candidats esclaves]
    reorder_map = np.concatenate([protected_local, free_to_slave_local])
    n_protected = len(protected_local)

    # Extraire la sous-matrice cinématique
    K_sub = K_cinematique[np.ix_(kinematic_constrained_indices, kinematic_constrained_indices)]

    # Appliquer la réorganisation (lignes ET colonnes)
    K_reordered = K_sub[np.ix_(reorder_map, reorder_map)]

    return K_reordered, reorder_map, n_protected


def QR_inverse_double_permutation(P, reorder_map, rank):
    """
    Calcule le mapping des maîtres depuis l'espace QR vers l'espace kinematic original.

    Cette fonction gère le double mapping :
    1. P^-1 : dé-pivotage QR
    2. reorder_map^-1 : dé-réorganisation initiale

    Parameters
    ----------
    P : ndarray
        Permutation retournée par QR (indices dans l'espace réordonné)
    reorder_map : ndarray
        Permutation initiale appliquée avant QR
    rank : int
        Nombre de DDL maîtres

    Returns
    -------
    master_in_kinematic : ndarray
        Indices des maîtres dans l'espace kinematic_constrained original
    combined_permutation : ndarray
        Permutation composée pour T_c_full
    """
    # Les maîtres dans l'espace QR (après pivotage)
    master_in_reordered = P[:rank]

    # Créer l'inverse de reorder_map
    inverse_reorder = np.empty_like(reorder_map)
    inverse_reorder[reorder_map] = np.arange(len(reorder_map))

    # Mapper vers l'espace kinematic original
    master_in_kinematic = inverse_reorder[master_in_reordered]

    # Permutation composée pour la construction directe de T_c_full
    combined_permutation = reorder_map[P]

    return master_in_kinematic, combined_permutation


def QR_validate_protected_masters(protected_global, master_indices_global, raise_error = True):
    """
    Vérifie que tous les DDL protégés sont bien devenus maîtres.

    Parameters
    ----------
    protected_global : ndarray
        Indices globaux des DDL devant être protégés
    master_indices_global : ndarray
        Indices globaux des DDL maîtres après QR

    Raises
    ------
    RuntimeError
        Si un DDL protégé n'est pas maître
    """
    not_protected = [i for i in protected_global if i not in master_indices_global]

    if len(not_protected) > 0:
        message = (f"ERREUR CRITIQUE : Des DDL protégés ne sont pas maîtres !\n"
            f"DDL protégés manquants : {not_protected}\n"
            f"Cela indique un bug dans l'algorithme de réorganisation.")


        if raise_error :
            raise RuntimeError(message)
        else :
            warnings.warn(message)


def QR_identify_protected_dof(fixeddof, kinematic_constrained_indices,
                              body_index_map : dict,
                              protected_dof_spec = None,
                              ):
    """
    Identifie les DDL fixés qui sont contraints cinématiquement et doivent être protégés.

    Parameters
    ----------
    fixeddof : array-like
        Indices globaux des DDL fixés (excitations du système)
    kinematic_constrained_indices : array-like
        Indices globaux des DDL contraints cinématiquement

    Returns
    -------
    protected_global : ndarray
        Indices globaux des DDL à protéger
    protected_local : ndarray
        Indices locaux (dans kinematic_constrained_indices) des DDL à protéger
    """

    if protected_dof_spec is not None :
        protected_global = []
        for (body_name, axe_index) in protected_dof_spec :
            if body_name not in body_index_map :
                raise IndexError(f"Le corps '{body_name}' n'est pas parmis les corps de référence.")
            if axe_index<0 or axe_index>=6 :
                raise IndexError(f"L'axe '{axe_index}' n'est pas valide.")

            dof = 6 * body_index_map[body_name] + axe_index
            if (dof not in protected_global) and (dof in kinematic_constrained_indices) :
                protected_global.append(dof)

    else :
        # Intersection : DDL fixés ET cinématiquement contraints
        protected_global = np.array([i for i in fixeddof if i in kinematic_constrained_indices])

    # Mapping global → local dans kinematic_constrained_indices
    kinematic_dict = {val: idx for idx, val in enumerate(kinematic_constrained_indices)}
    protected_local = np.array([kinematic_dict[i] for i in protected_global])

    return protected_global, protected_local


def QR_create_reordered_matrix(K_cinematique, kinematic_constrained_indices, protected_local):
    """
    Réorganise la matrice cinématique pour placer les DDL protégés en premier.

    Parameters
    ----------
    K_cinematique : ndarray
        Matrice de raideur cinématique complète (ndof_total × ndof_total)
    kinematic_constrained_indices : ndarray
        Indices globaux des DDL contraints cinématiquement
    protected_local : ndarray
        Indices locaux des DDL à protéger

    Returns
    -------
    K_reordered : ndarray
        Matrice réorganisée (ndof_qr × ndof_qr)
    reorder_map : ndarray
        Permutation appliquée : [protected_local, free_to_slave_local]
    n_protected : int
        Nombre de DDL protégés
    """
    ndof_qr = len(kinematic_constrained_indices)

    # Séparer protected et free_to_slave
    all_local_indices = np.arange(ndof_qr)
    free_to_slave_local = np.array([i for i in all_local_indices if i not in protected_local])

    # Créer la permutation : [protected en tête | candidats esclaves]
    reorder_map = np.concatenate([protected_local, free_to_slave_local])
    n_protected = len(protected_local)

    # Extraire la sous-matrice cinématique
    K_sub = K_cinematique[np.ix_(kinematic_constrained_indices, kinematic_constrained_indices)]

    # Appliquer la réorganisation (lignes ET colonnes)
    K_reordered = K_sub[np.ix_(reorder_map, reorder_map)]

    return K_reordered, reorder_map, n_protected


def QR_protectedPivoting(K_reordered, n_protected, rtol=None):
    n, m = K_reordered.shape
    eps = np.finfo(float).eps

    if n_protected == 0:
        Q, R, P = QR_decomposition(K_reordered, mode='full', pivoting=True)
        svd = np.linalg.svd(K_reordered, compute_uv=False)
        tolerance = (rtol if rtol is not None else np.sqrt(eps) * max(n, m)) * svd.max()
        rank = np.sum(svd > tolerance)
        return Q, R, P, rank

    # ÉTAPE 1 : QR sur colonnes protégées
    K_protected = K_reordered[:, :n_protected]
    svd_protected = np.linalg.svd(K_protected, compute_uv=False)
    tol_protected = (rtol if rtol is not None else np.sqrt(eps) * max(K_protected.shape)) * svd_protected.max()
    rank_protected = np.sum(svd_protected > tol_protected)

    if rank_protected < n_protected:
        raise ValueError(
            f"Les colonnes protégées sont linéairement dépendantes !\n"
            f"Rang détecté : {rank_protected}/{n_protected}"
        )

    Q1, R1 = QR_decomposition(K_protected, mode='economic', pivoting=False)
    # Q1 : (n, n_protected), R1 : (n_protected, n_protected)

    if m > n_protected:
        # ÉTAPE 2 : Projection des colonnes restantes
        K_remaining = K_reordered[:, n_protected:]
        R12 = Q1.T @ K_remaining
        # R12 : (n_protected, m - n_protected)

        # ÉTAPE 3 : Composante orthogonale
        K_ortho = K_remaining - Q1 @ R12
        # K_ortho : (n, m - n_protected)

        # ÉTAPE 4 : QR sur composante orthogonale avec pivotage
        Q2, R22, P2 = QR_decomposition(K_ortho, mode='economic', pivoting=True)
        # Q2 : (n, rank_remaining), R22 : (rank_remaining, m - n_protected)

        # Détection du rang
        svd_remaining = np.linalg.svd(K_ortho, compute_uv=False)
        tol_remaining = (rtol if rtol is not None else np.sqrt(eps) * max(K_ortho.shape)) * svd_remaining.max()
        rank_remaining = np.sum(svd_remaining > tol_remaining)

        # Rang total
        rank = n_protected + rank_remaining

        # ÉTAPE 5 : Assemblage
        # Q_global : (n, rank)
        Q_global = np.hstack([Q1, Q2[:, :rank_remaining]])

        # R_global : (rank, m)
        R_global = np.zeros((rank, m))
        R_global[:n_protected, :n_protected] = R1
        R_global[:n_protected, n_protected:] = R12[:, P2]  # Appliquer permutation
        R_global[n_protected:rank, n_protected:] = R22[:rank_remaining, :]

        # Permutation globale
        P_global = np.arange(m)
        P_global[n_protected:] = P2 + n_protected

    else:
        # Toutes les colonnes sont protégées
        Q_global = Q1
        R_global = R1
        P_global = np.arange(m)
        rank = n_protected

    return Q_global, R_global, P_global, rank