#!/usr/bin/env julia
"""
Tensor contraction examples
"""

using Random
using TensorOperations

function main()
    println("matrix-matrix multiplication")
    A = rand(2, 3)
    B = rand(3, 4)
    println("A: shape ", size(A), "\n", A, "\n")
    println("B: shape ", size(B), "\n", B, "\n")
    println("contract: A, B -> C")
    @tensor C[i, k] := A[i, j] * B[j, k]
    println("C: shape ", size(C), "\n", C, "\n")

    println("more complex contraction")
    A = rand(2, 3, 4, 5)
    B = rand(4, 3)
    C = rand(5, 3, 4)
    println("A: shape ", size(A), "\n", A, "\n")
    println("B: shape ", size(B), "\n", B, "\n")
    println("C: shape ", size(C), "\n", C, "\n")
    println("contract: A, B, C -> D")
    @tensor D[i, j, k] := A[i, j, l, m] * B[l, n] * C[m, n, k]
    println("D: shape ", size(D), "\n", D)
end

main()
