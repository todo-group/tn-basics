#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tensor definition examples
"""

import numpy as np


def main():
    """tensor definition"""
    print("vector (1-leg tensor)")
    t1 = np.array([1, 2, 3])
    print(f"t1 = {t1}")
    print(f"t1.shape = {t1.shape}")
    print(f"t1[0] = {t1[0]}")
    print(f"t1[1] = {t1[1]}\n")

    print("(random) matrix (2-leg tensor)")
    t2 = np.random.rand(2, 3)
    print(f"t2 =\n{t2}")
    print(f"t2.shape = {t2.shape}")
    print(f"t2[0,0] = {t2[0, 0]}")
    print(f"t2[1,2] = {t2[1, 2]}\n")

    print("3D array (3-leg tensor)")
    A = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print(f"A =\n{A}\n")
    print(f"A.shape = {A.shape}")
    print(f"A[0,0,0] = {A[0, 0, 0]}")
    print(f"A[1,1,1] = {A[1, 1, 1]}")
    print(f"A[:,0,1] = {A[:, 0, 1]}\n")


if __name__ == "__main__":
    main()
