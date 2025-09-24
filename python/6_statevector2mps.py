#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate MPS from statevector
"""

import numpy as np


def main():
    cutoff = 1e-10

    # Bell state
    print("Bell state:")
    v = np.array([1, 0, 0, 1]) / np.sqrt(2)

    v = v.reshape((2, 2))
    U, S, Vt = np.linalg.svd(v, full_matrices=False)
    print(f"singular values: {S}")
    S = S[S > cutoff * S[0]]
    rank_new = len(S)
    U = U[:, :rank_new]
    Vt = Vt[:rank_new, :]
    v = np.dot(np.diag(S), Vt)
    print(f"tensors {[U, v]})\n")

    # GHZ state
    n = 6
    print(f"n={n} GHZ state:")
    v = np.zeros(2**n)
    v[0] = 1 / np.sqrt(2)
    v[-1] = 1 / np.sqrt(2)

    mps = []
    rank = 1
    for i in range(0, n - 1):
        v = v.reshape((rank * 2, -1))
        U, S, Vt = np.linalg.svd(v, full_matrices=False)
        print(f"{i}: singular values: {S}")
        S = S[S > cutoff * S[0]]
        rank_new = len(S)
        U = U[:, :rank_new]
        Vt = Vt[:rank_new, :]
        if i > 0:
            U = U.reshape((rank, 2, rank_new))
        mps.append(U)
        v = np.dot(np.diag(S), Vt)
        rank = rank_new
    v = v.reshape((rank, 2))
    mps.append(v)
    print(f"tensors: {mps}\n")


if __name__ == "__main__":
    main()
