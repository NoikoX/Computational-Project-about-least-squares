import numpy as np


def classical_gram_schmidtQR(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j]

        for i in range(j):
            R[i, j] = Q[:, i].T @ A[:, j]
            v = v - (R[i, j] * Q[:, i])

        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R


def modified_gram_schmidt(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    V = A.copy()

    for i in range(n):
        R[i, i] = np.linalg.norm(V[:, i])
        Q[:, i] = V[:, i] / R[i, i]

        for j in range(i + 1, n):
            R[i, j] = Q[:, i].T @ V[:, j]
            V[:, j] = V[:, j] - R[i, j] * Q[:, i]

    return Q, R


def least_sqrs_with_inv(A, b):
    return np.linalg.inv((A.T @ A)) @ A.T @ b


def least_sqrs_with_QR(A, b):
    Q, R = classical_gram_schmidtQR(A)

    c = Q.T @ b
    return np.linalg.inv(R) @ c


# A = np.array([
#     [2, 12],
#     [-2, -6],
#     [1, 0]
# ])
# Q, R = classical_gram_schmidtQR(A)
# print(Q,"\n", R)
# Q, R = modified_gram_schmidt(A)
# print(Q,"\n", R)