#!/usr/bin/env julia
"""
Tensor definition examples
"""

using Random

function main()
    println("vector (1-leg tensor)")
    t1 = [1, 2, 3]
    println("t1 = $t1")
    println("t1.shape = ", size(t1))
    println("t1[1] = ", t1[1])   # Julia は 1-based index
    println("t1[2] = ", t1[2], "\n")

    println("(random) matrix (2-leg tensor)")
    t2 = rand(2, 3)
    println("t2 =\n", t2)
    println("t2.shape = ", size(t2))
    println("t2[1,1] = ", t2[1, 1])
    println("t2[2,3] = ", t2[2, 3], "\n")

    println("3D array (3-leg tensor)")
    A = reshape([1, 2, 3, 4, 5, 6, 7, 8], 2, 2, 2)
    A = permutedims(A, (3, 2, 1))
    println("A =\n", A, "\n")
    println("A.shape = ", size(A))
    println("A[1,1,1] = ", A[1, 1, 1])
    println("A[2,2,1] = ", A[2, 2, 1])
    println("A[:,1,2] = ", A[:, 1, 2])
end

main()
