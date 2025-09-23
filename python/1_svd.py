#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVD and low-rank approximation of a matrix
"""

import numpy as np


def main():
    A = np.array([[1, 2, 3], [6, 4, 5], [8, 9, 7], [10, 11, 12]])
    print(f"A =\n{A}\n")

    """(thin) SVD"""
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    print(f"U =\n{U}\n")
    print(f"S =\n{S}\n")
    print(f"Vt =\n{Vt}\n")

    """reconstruct A"""
    S_matrix = np.diag(S)
    A_reconstructed = U @ S_matrix @ Vt

    print(f"Reconstructed A =\n{A_reconstructed}\n")

    """full SVD"""
    U, S, Vt = np.linalg.svd(A, full_matrices=True)

    print(f"U (full) =\n{U}\n")
    print(f"S (full) =\n{S}\n")
    print(f"Vt (full) =\n{Vt}\n")

    """reconstruct A"""
    S_matrix = np.zeros((4, 3))
    np.fill_diagonal(S_matrix, S)
    A_reconstructed = U @ S_matrix @ Vt

    print(f"Reconstructed A =\n{A_reconstructed}\n")

    """rank-2 approximation"""
    r = 2
    Ur = U[:, :r]
    Sr = np.diag(S[:r])
    Vtr = Vt[:r, :]
    A_rank2 = Ur @ Sr @ Vtr

    print(f"Rank-2 approximation of A =\n{A_rank2}\n")
    print(f"Frobenius norm of the error = {np.linalg.norm(A - A_rank2, 'fro')}\n")


if __name__ == "__main__":
    main()
