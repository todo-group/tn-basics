#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QTT representation of finite-difference operator
"""

import matplotlib.pyplot as plt
import numpy as np


def main():
    depth = 4
    npoints = 2**depth
    cutoff = 1e-10
    max_rank = 4

    """target function and its derivative"""
    x = np.linspace(0.0, 1.0, npoints)
    y = np.exp(x)
    dy = np.exp(x)
    # y = np.cos(x)
    # dy = -np.sin(x)
    # y = np.exp(-(((x - 0.5) / 0.1) ** 2)) / (0.1 * np.sqrt(np.pi))
    # dy = -2.0 * (x - 0.5) * y / (0.1**2)

    """QTT decomposition of target function"""
    yt = y.copy()
    qtt = []
    rank = 1
    for k in range(0, depth - 1):
        yt = yt.reshape((rank * 2, -1))
        U, S, Vt = np.linalg.svd(yt, full_matrices=False)
        S = S[S > cutoff * S[0]]
        S = S[: min(len(S), max_rank)]
        rank_new = len(S)
        U = U[:, :rank_new]
        Vt = Vt[:rank_new, :]
        U = U.reshape((rank, 2, rank_new))
        qtt.append(U)
        yt = np.dot(np.diag(S), Vt)
        rank = rank_new
    yt = yt.reshape((rank, 2, 1))
    qtt.append(yt)
    print(f"QTT virtual dimensions: {[qtt[k].shape for k in range(depth)]}\n")

    """QTT for index shift operator"""
    s_qtt = []
    s = np.zeros((1, 2, 2, 2))
    s[0, 0, 0, 0] = s[0, 1, 1, 0] = s[0, 1, 0, 1] = s[0, 0, 1, 1] = 1.0
    s_qtt.append(s)
    for k in range(1, depth - 1):
        s = np.zeros((2, 2, 2, 2))
        s[0, 0, 0, 0] = s[0, 1, 1, 0] = s[0, 1, 0, 1] = s[1, 0, 1, 1] = 1.0
        s_qtt.append(s)
    s = np.zeros((2, 2, 2, 1))
    s[0, 1, 0, 0] = s[1, 0, 1, 0] = 1.0
    s_qtt.append(s)
    print(
        f"index shift operator virtual dimensions: {[s_qtt[k].shape for k in range(depth)]}\n"
    )

    """check for index shift operator"""
    if depth < 5:
        s_op = s_qtt[0]
        s_op = s_op.reshape(2, 2, 2)
        for k in range(1, depth):
            s_op = np.einsum("ijk,klmn->iljmn", s_op, s_qtt[k])
            s_op = s_op.reshape(
                s_op.shape[0] * s_op.shape[1],
                s_op.shape[2] * s_op.shape[3],
                s_op.shape[4],
            )
        s_op = s_op.reshape(s_op.shape[0], s_op.shape[1])
        print(f"index shift operator:\n{s_op}\n")

    """QTT for finite-difference operator"""
    d_qtt = []
    s = s_qtt[0].copy()
    st = s.transpose((0, 2, 1, 3))
    d = np.zeros((s.shape[0], 2, 2, 2 * s.shape[3]))
    d[:, :, :, : s.shape[3]] = s
    d[:, :, :, s.shape[3] : 2 * s.shape[3]] = -st  # minus sign for derivative
    d_qtt.append(d)
    for k in range(1, depth - 1):
        s = s_qtt[k].copy()
        st = s.transpose((0, 2, 1, 3))
        d = np.zeros((2 * s.shape[0], 2, 2, 2 * s.shape[3]))
        d[: s.shape[0], :, :, : s.shape[3]] = 2 * s
        d[s.shape[0] : 2 * s.shape[0], :, :, s.shape[3] : 2 * s.shape[3]] = 2 * st
        d_qtt.append(d)
    s = s_qtt[depth - 1].copy()
    st = s.transpose((0, 2, 1, 3))
    d = np.zeros((2 * s.shape[0], 2, 2, s.shape[3]))
    d[: s.shape[0], :, :, :] = 2 * s
    d[s.shape[0] : 2 * s.shape[0], :, :, :] = 2 * st
    d_qtt.append(d)
    print(
        f"finite-difference operator virtual dimensions: {[d_qtt[k].shape for k in range(depth)]}\n"
    )

    """check for finite-difference operator"""
    if depth < 5:
        d_op = d_qtt[0]
        d_op = d_op.reshape(2, 2, 4)
        for k in range(1, depth):
            d_op = np.einsum("ijk,klmn->iljmn", d_op, d_qtt[k])
            d_op = d_op.reshape(
                d_op.shape[0] * d_op.shape[1],
                d_op.shape[2] * d_op.shape[3],
                d_op.shape[4],
            )
        d_op = d_op.reshape(d_op.shape[0], d_op.shape[1])
        print(f"finite-difference operator:\n{d_op}\n")

    """apply finite-difference operator to target function"""
    dy_qtt = []
    for k in range(depth):
        t = np.einsum("ijkl,mjn->miknl", d_qtt[k], qtt[k])
        t = t.reshape(t.shape[0] * t.shape[1], t.shape[2], t.shape[3] * t.shape[4])
        dy_qtt.append(t)
    print(
        f"derivative QTT virtual dimensions: {[dy_qtt[k].shape for k in range(depth)]}\n"
    )

    """reconstruction"""
    dyr = dy_qtt[0]
    dyr = dyr.reshape(dyr.shape[0] * dyr.shape[1], dyr.shape[2])
    for k in range(1, depth):
        dyr = np.einsum("ij,jkl->ikl", dyr, dy_qtt[k])
        dyr = dyr.reshape(dyr.shape[0] * dyr.shape[1], dyr.shape[2])
    dyr = dyr.reshape(-1)

    """drop boundary points as derivative is not correct there due to boundary conditions"""
    x = x[1:-2]
    dy = dy[1:-2]
    dyr = dyr[1:-2]
    print(f"target derivative y' = {dy}\n")
    print(f"QTT derivative y' = {dyr}\n")
    print(f"error = {np.linalg.norm(dy - dyr) / len(dy)}\n")

    plt.figure()
    plt.plot(x, dy, label="target")
    plt.plot(x, dyr, label="QTT")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(x, dy - dyr, label="error")
    plt.title("error")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
