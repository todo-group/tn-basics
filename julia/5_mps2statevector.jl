#!/usr/bin/env julia
"""
Generate statevector from MPS
"""

using TensorOperations

function main()
    # Bell state
    println("Bell state:")
    tl = [1.0 0.0; 0.0 1.0] / 2^(0.25)
    tr = [1.0 0.0; 0.0 1.0] / 2^(0.25)
    println("left tensor: ", tl)
    println("right tensor: ", tr, "\n")

    @tensor bell[i, k] := tl[i, j] * tr[j, k]
    bell = vec(permutedims(bell))
    println("statevector: ", bell, "\n")

    # GHZ state
    n = 6
    println("n=$(n) GHZ state:")
    w = 2^(1 / (2n))
    tl = [1.0 0.0; 0.0 1.0] / w
    tr = [1.0 0.0; 0.0 1.0] / w
    t = zeros(Float64, 2, 2, 2)
    t[1, 1, 1] = 1 / w
    t[2, 2, 2] = 1 / w
    println("left tensor: ", tl)
    println("right tensor: ", tr)
    println("middle tensors: ", t, "\n")

    ghz = tl
    for _ = 2:(n-1)
        @tensor tmp[i, k, l] := ghz[i, j] * t[j, k, l]
        ghz = reshape(tmp, size(tmp, 1) * size(tmp, 2), size(tmp, 3)) # (i*k, l)
    end
    @tensor ghz[i, k] := ghz[i, j] * tr[j, k]
    ghz = vec(permutedims(ghz))
    println("statevector: ", ghz)
end

main()
