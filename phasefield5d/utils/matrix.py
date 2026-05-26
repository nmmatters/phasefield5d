import numpy as np

# Orthogonalization matrix for 5-component Gibbs simplex (maps 4-comp reduced space to full)
orthogonalization_matrix = np.array([
    [-1,          -1/np.sqrt(3), -1/np.sqrt(6),  -1/np.sqrt(10)],
    [ 1,          -1/np.sqrt(3), -1/np.sqrt(6),  -1/np.sqrt(10)],
    [ 0,           2/np.sqrt(3), -1/np.sqrt(6),  -1/np.sqrt(10)],
    [ 0,           0,             np.sqrt(3/2),  -1/np.sqrt(10)],
])

gram_matrix = orthogonalization_matrix.T @ orthogonalization_matrix


def transform_to_full_element_vector(vector):
    """Append Fe = 1 - sum(others) to a 4-component composition vector."""
    return np.concatenate([[1 - vector.sum()], vector])


def transform_to_full_eigenvector(vector):
    """Append Fe component (= -sum) to a 4-component eigenvector."""
    return np.concatenate([[-vector.sum()], vector])


def transform_to_full_element_vector_array(X_array):
    """Broadcast version of transform_to_full_element_vector for arrays (..., 4) -> (..., 5)."""
    x0 = 1.0 - np.sum(X_array, axis=-1, keepdims=True)
    return np.concatenate([x0, X_array], axis=-1)


def align_vector(vector):
    """Flip sign so the largest-magnitude component is positive."""
    if abs(vector.min()) > abs(vector.max()):
        vector = -vector
    return vector


def orthogonalize_single_matrix(matrix, O=None):
    if O is None:
        O = orthogonalization_matrix
    if matrix is None:
        return None
    N = matrix.shape[0]
    O = O[:N, :N]
    return O.T @ matrix @ O


def orthogonalize_single_vector(vector, O=None):
    """Gibbs -> Cartesian."""
    if O is None:
        O = orthogonalization_matrix
    return np.linalg.inv(O) @ vector


def re_orthogonalize_single_vector(vector, O=None):
    """Cartesian -> Gibbs."""
    if O is None:
        O = orthogonalization_matrix
    return O @ vector


def normalize_in_metric(v, G):
    v = np.asarray(v).reshape(-1, 1)
    return (v / np.sqrt(float(v.T @ G @ v))).ravel()


def calculate_eigenvalues_and_eigenvectors(matrix):
    return np.linalg.eigh(matrix)


def calculate_effective_projection(vector, matrix):
    return np.dot(vector.T, matrix @ vector)


def is_symmetric(A, rtol=1e-10, atol=1e-12):
    return np.allclose(A, A.T, rtol=rtol, atol=atol)


def is_spd(A, sym_check=True, tol=1e-12):
    if sym_check and not is_symmetric(A):
        return False
    return np.linalg.eigvalsh(A).min() > tol


def is_psd(A, sym_check=True, tol=1e-12):
    if sym_check and not is_symmetric(A):
        return False
    return np.linalg.eigvalsh(A).min() >= -tol
