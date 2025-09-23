#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compress and reconstruct grayscale images using SVD
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def main():
    path = "../data/sqai-square-gray-rgb150ppi.jpg"  # path to input image

    """load image and convert to grayscale"""
    image = Image.open(path).convert("L")
    array = np.asarray(image, dtype=np.float64)
    h, w = array.shape
    print(f"image size; {h} {w}\n")
    plt.figure()
    plt.imshow(array, cmap="gray", vmin=0, vmax=255)
    plt.title("original image")
    plt.axis("off")
    plt.show()

    """SVD"""
    U, S, Vt = np.linalg.svd(array, full_matrices=False)
    plt.figure()
    plt.semilogy(S)
    plt.title("singular values")
    plt.xlabel("index")
    plt.ylabel("λ_i")
    plt.show()

    """image reconstruction with different ranks"""
    ranks = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    for r in ranks:
        Ur = U[:, :r]
        Sr = np.diag(S[:r])
        Vtr = Vt[:r, :]
        Ar = Ur @ Sr @ Vtr
        plt.figure()
        plt.imshow(Ar, cmap="gray", vmin=0, vmax=255)
        plt.title(f"reconstructed image (rank {r})")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
