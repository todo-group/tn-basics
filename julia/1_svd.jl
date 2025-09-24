#!/usr/bin/env julia
# -*- coding: utf-8 -*-
#
# SVD and low-rank approximation of a matrix

using LinearAlgebra
using Printf

function main()
    A = [1 2 3;
        6 4 5;
        8 9 7;
        10 11 12]
    println("A =\n", A, "\n")

    # (thin) SVD
    F = svd(A; full=false)
    U, S, V = F.U, F.S, F.V
    println("U =\n", U, "\n")
    println("S =\n", S, "\n")
    println("Vt =\n", transpose(V), "\n")

    # reconstruct A
    S_matrix = Diagonal(S)
    A_reconstructed = U * S_matrix * transpose(V)
    println("Reconstructed A =\n", A_reconstructed, "\n")

    # full SVD
    Ffull = svd(A; full=true)
    Ufull, Sfull, Vfull = Ffull.U, Ffull.S, Ffull.V
    println("U (full) =\n", Ufull, "\n")
    println("S (full) =\n", Sfull, "\n")
    println("Vt (full) =\n", transpose(Vfull), "\n")

    # reconstruct A (full SVD)
    S_matrix_full = zeros(eltype(Sfull), size(A)...)
    for i in eachindex(Sfull)
        S_matrix_full[i, i] = Sfull[i]
    end
    A_reconstructed_full = Ufull * S_matrix_full * transpose(Vfull)
    println("Reconstructed A =\n", A_reconstructed_full, "\n")

    # rank-2 approximation
    r = 2
    Ur = Ufull[:, 1:r]
    Sr = Diagonal(Sfull[1:r])
    Vr = Vfull[:, 1:r]
    A_rank2 = Ur * Sr * transpose(Vr)
    println("Rank-2 approximation of A =\n", A_rank2, "\n")

    err = norm(A .- A_rank2)
    @printf "Frobenius norm of the error = %.10f\n" err
end

main()
