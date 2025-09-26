#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(Simplified) TEBD simulation of random quantum circuits
"""

import numpy as np


def main():
    n = 16  # number of qubits
    depth = 16  # number of layers
    max_dim = 4  # for MPS with truncation
    cutoff = 1e-10  # for MPS with truncation

    # two-qubit gate: CNOT
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]).reshape(
        (2, 2, 2, 2)
    )

    # state vector |0...0>
    state = np.zeros((2**n, 1))
    state[0] = 1
    state = state.reshape((2,) * n)

    # MPS |0...0> without truncation
    mps0 = []
    for i in range(n):
        mps0.append(np.array([[[1], [0]]]))
    print(
        f"mps0/1 initial virtual bond dimensions: {[mps0[i].shape[2] for i in range(n - 1)]}"
    )

    # MPS |0...0> with truncation
    mps1 = mps0.copy()

    for k in range(depth):
        # random single-qubit rotations
        for pos in range(n):
            alpha = np.random.uniform(0, 2 * np.pi)
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2 * np.pi)
            U = np.exp(1j * alpha) * np.array(
                [
                    [np.cos(theta / 2), -np.exp(1j * phi) * np.sin(theta / 2)],
                    [np.exp(-1j * phi) * np.sin(theta / 2), np.cos(theta / 2)],
                ]
            )

            ss_op = list(range(2))
            ss_in = list(range(2, 2 + n))
            ss_to = list(range(2, 2 + n))
            ss_in[pos] = ss_op[1]
            ss_to[pos] = ss_op[0]
            state = np.einsum(U, ss_op, state, ss_in, ss_to)

            mps0[pos] = np.einsum("ij,kjm->kim", U, mps0[pos])
            mps1[pos] = np.einsum("ij,kjm->kim", U, mps1[pos])

        # cnot gates
        for i in range(k // 2, n - 1, 2):
            pos = [i, i + 1]

            ss_op = list(range(4))
            ss_in = list(range(4, 4 + n))
            ss_to = list(range(4, 4 + n))
            ss_in[pos[0]] = ss_op[2]
            ss_in[pos[1]] = ss_op[3]
            ss_to[pos[0]] = ss_op[0]
            ss_to[pos[1]] = ss_op[1]
            state = np.einsum(cnot, ss_op, state, ss_in, ss_to)

            t = np.einsum("ijkl,mkn->mijln", cnot, mps0[pos[0]])
            t = np.einsum("ijklm,mln->ijkn", t, mps0[pos[1]])
            t = t.reshape(t.shape[0] * t.shape[1], t.shape[2] * t.shape[3])
            U, S, Vt = np.linalg.svd(t, full_matrices=False)
            S = S[S > cutoff]
            S_sqrt = np.diag(np.sqrt(S))
            mps0[pos[0]] = (U[:, : len(S)] @ S_sqrt).reshape(-1, 2, len(S))
            mps0[pos[1]] = (S_sqrt @ Vt[: len(S), :]).reshape(len(S), 2, -1)

            t = np.einsum("ijkl,mkn->mijln", cnot, mps1[pos[0]])
            t = np.einsum("ijklm,mln->ijkn", t, mps1[pos[1]])
            t = t.reshape(t.shape[0] * t.shape[1], t.shape[2] * t.shape[3])
            U, S, Vt = np.linalg.svd(t, full_matrices=False)
            S = S[S > cutoff]
            S = S[: min(len(S), max_dim)]
            S_sqrt = np.diag(np.sqrt(S))
            mps1[pos[0]] = (U[:, : len(S)] @ S_sqrt).reshape(-1, 2, len(S))
            mps1[pos[1]] = (S_sqrt @ Vt[: len(S), :]).reshape(len(S), 2, -1)

        state0 = mps0[0].reshape(2, -1)
        for i in range(1, n):
            state0 = np.einsum("ij,jkl->ikl", state0, mps0[i])
            state0 = state0.reshape(state0.shape[0] * state0.shape[1], state0.shape[2])
        state0 = state0.reshape(-1)

        state1 = mps1[0].reshape(2, -1)
        for i in range(1, n):
            state1 = np.einsum("ij,jkl->ikl", state1, mps1[i])
            state1 = state1.reshape(state1.shape[0] * state1.shape[1], state1.shape[2])
        state1 = state1.reshape(-1)

        print(
            f"step: {k} fidelity: {np.abs(np.vdot(state, state0)) ** 2}, {np.abs(np.vdot(state, state1)) ** 2}"
        )

    print(
        f"mps0 final virtual bond dimensions: {[mps0[i].shape[2] for i in range(n - 1)]}"
    )
    print(
        f"mps1 final virtual bond dimensions: {[mps1[i].shape[2] for i in range(n - 1)]}"
    )


if __name__ == "__main__":
    main()
