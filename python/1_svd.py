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
    print(f"reconstructed A =\n{A_reconstructed}\n")

    """full SVD"""
    Ufull, Sfull, Vtfull = np.linalg.svd(A, full_matrices=True)
    print(f"U (full) =\n{Ufull}\n")
    print(f"S (full) =\n{Sfull}\n")
    print(f"Vt (full) =\n{Vtfull}\n")

    """reconstruct A (full SVD)"""
    S_matrix_full = np.zeros((4, 3))
    np.fill_diagonal(S_matrix, Sfull)
    A_reconstructed_full = Ufull @ S_matrix_full @ Vtfull
    print(f"reconstructed A =\n{A_reconstructed_full}\n")

    """rank-2 approximation"""
    r = 2
    Ur = U[:, :r]
    Sr = np.diag(S[:r])
    Vtr = Vt[:r, :]
    A_rank2 = Ur @ Sr @ Vtr

    print(f"rank-2 approximation of A =\n{A_rank2}\n")
    print(f"Frobenius norm of the error = {np.linalg.norm(A - A_rank2, 'fro')}")


if __name__ == "__main__":
    main()
