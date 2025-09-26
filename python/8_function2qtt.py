#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conversion of one-dimensional function into QTT representation
"""

import matplotlib.pyplot as plt
import numpy as np


def main():
    depth = 8
    npoints = 2**depth
    cutoff = 1e-10
    max_rank = 4

    """target function"""
    x = np.linspace(0.0, 1.0, npoints)
    y = np.exp(x)
    # y = np.sin(x)
    # y = np.exp(-(((x - 0.5) / 0.1) ** 2)) / (0.1 * np.sqrt(np.pi))
    if depth < 5:
        print(f"x = {x}")
        print(f"y = {y}\n")

    """QTT decomposition"""
    yt = y.copy()
    qtt = []
    rank = 1
    for k in range(0, depth - 1):
        print(f"depth: {k}")
        yt = yt.reshape((rank * 2, -1))
        U, S, Vt = np.linalg.svd(yt, full_matrices=False)
        print(f"singular values: {S}")
        S = S[S > cutoff * S[0]]
        S = S[: min(len(S), max_rank)]
        rank_new = len(S)
        U = U[:, :rank_new]
        Vt = Vt[:rank_new, :]
        if k > 0:
            U = U.reshape((rank, 2, rank_new))
        print(f"tensor shape: {U.shape}\n")
        qtt.append(U)
        yt = np.dot(np.diag(S), Vt)
        rank = rank_new
    yt = yt.reshape((rank, 2))
    print(f"depth: {depth - 1}")
    print(f"tensor shape: {yt.shape}\n")
    qtt.append(yt)
    if depth < 5:
        print(f"qtt: {qtt}\n")

    yr = qtt[0]
    for k in range(1, depth - 1):
        yr = np.einsum("ij,jkl->ikl", yr, qtt[k])
        yr = yr.reshape(yr.shape[0] * yr.shape[1], yr.shape[2])
    yr = np.einsum("ij,jk->ik", yr, qtt[depth - 1])
    yr = yr.reshape(-1)
    print(f"error = {np.linalg.norm(y - yr) / npoints}\n")

    if depth < 5:
        print(f"reconstructed y = {yr}\n")
    plt.figure()
    plt.plot(x, y, label="target")
    if npoints <= 32:
        plt.plot(x, yr, "o", label="QTT")
    else:
        plt.plot(x[:: npoints // 32], yr[:: npoints // 32], "o", label="QTT")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
