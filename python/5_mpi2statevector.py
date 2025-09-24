#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate statevector from MPS
"""

import numpy as np


def main():
    # Bell state
    print("Bell state:")
    tl = np.array([[1, 0], [0, 1]]) / np.pow(2, 0.25)
    tr = np.array([[1, 0], [0, 1]]) / np.pow(2, 0.25)
    print(f"left tensor: {tl}")
    print(f"right tensor: {tr}\n")

    bell = np.einsum("ij,jk->ik", tl, tr)
    bell = bell.reshape(-1)
    print(f"statevector: {bell}\n")

    # GHZ state
    n = 6
    print(f"n={n} GHZ state:")
    w = np.pow(2, 1 / (2 * n))
    tl = np.array([[1, 0], [0, 1]]) / w
    tr = np.array([[1, 0], [0, 1]]) / w
    t = np.zeros((2, 2, 2))
    t[0, 0, 0] = 1 / w
    t[1, 1, 1] = 1 / w
    print(f"left tensor: {tl}")
    print(f"right tensor: {tr}")
    print(f"middle tensors: {t}\n")

    ghz = tl
    for k in range(1, n - 1):
        ghz = np.einsum("ij,jkl->ikl", ghz, t)
        ghz = ghz.reshape(ghz.shape[0] * ghz.shape[1], ghz.shape[2])
    ghz = np.einsum("ij,jk->ik", ghz, tr)
    ghz = ghz.reshape(-1)
    print(f"statevector: {ghz}\n")


if __name__ == "__main__":
    main()
