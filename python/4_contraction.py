#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tensor contraction examples
"""

import numpy as np


def main():
    print("matrix-matrix multiplication")
    A = np.random.random((2, 3))
    B = np.random.random((3, 4))
    print(f"A: shape {A.shape}\n{A}\n")
    print(f"B: shape {B.shape}\n{B}\n")
    print("contract: A, B -> C")
    C = np.einsum("ij,jk->ik", A, B)
    print(f"C: shape {C.shape}\n{C}\n")

    print("more complex contraction")
    A = np.random.random((2, 3, 4, 5))
    B = np.random.random((4, 3))
    C = np.random.random((5, 3, 4))
    print(f"A: shape {A.shape}\n{A}\n")
    print(f"B: shape {B.shape}\n{B}\n")
    print(f"C: shape {C.shape}\n{C}\n")
    print("contract: A, B, C -> D")
    D = np.einsum("ijlm,ln,mnk->ijk", A, B, C)
    print(f"D: shape {D.shape}\n{D}\n")


if __name__ == "__main__":
    main()
