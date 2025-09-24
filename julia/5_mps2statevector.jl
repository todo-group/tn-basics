#!/usr/bin/env julia
# -*- coding: utf-8 -*-
"""
Generate statevector from MPS (Julia)
"""

using TensorOperations

# Python の reshape(-1)（行優先）に合わせて 2D を一次元化する補助関数
cflatten(M::AbstractMatrix) = vec(permutedims(M))  # 行→列入替後に vec

function main()
    # Bell state
    println("Bell state:")
    tl = [1.0 0.0; 0.0 1.0] / 2^(0.25)
    tr = [1.0 0.0; 0.0 1.0] / 2^(0.25)
    println("left tensor: ", tl)
    println("right tensor: ", tr, "\n")

    @tensor bell[i, k] := tl[i, j] * tr[j, k]
    # Python の reshape(-1) と同順にしたい場合は cflatten を使う
    state_bell = cflatten(bell)
    println("statevector: ", state_bell, "\n")

    # GHZ state
    n = 6
    println("n=$(n) GHZ state:")
    w = 2^(1 / (2n))
    tl = [1.0 0.0; 0.0 1.0] / w
    tr = [1.0 0.0; 0.0 1.0] / w
    t = zeros(Float64, 2, 2, 2)  # (j,k,l)
    t[1, 1, 1] = 1 / w
    t[2, 2, 2] = 1 / w
    println("left tensor: ", tl)
    println("right tensor: ", tr)
    println("middle tensors: ", t, "\n")

    # 連結（縮約）: A(ij) と T(jkl) -> (i k l) にしてから (i*k, l) に reshape
    ghz = tl
    for _ = 2:(n-1)  # Python の range(1, n-1) と同じ回数 (n-2回)
        @tensor tmp[i, k, l] := ghz[i, j] * t[j, k, l]
        ghz = reshape(tmp, size(tmp, 1) * size(tmp, 2), size(tmp, 3)) # (i*k, l)
    end
    @tensor ghz2[i, k] := ghz[i, j] * tr[j, k]
    state_ghz = cflatten(ghz2)
    println("statevector: ", state_ghz, "\n")
end

main()
