#!/usr/bin/env julia
# -*- coding: utf-8 -*-
"""
Generate MPS from statevector (Julia)
"""
using LinearAlgebra

function truncate_svd(U::AbstractMatrix, S::AbstractVector, V::AbstractMatrix; cutoff=1e-10)
    # A ≈ U[:,1:r] * Diagonal(S[1:r]) * V[:,1:r]' だが、
    # Python 版と揃えるため v = diag(S) * V' の「左正準」っぽい形に
    r = sum(S .> cutoff * S[1])
    U2 = @view U[:, 1:r]
    S2 = @view S[1:r]
    Vt2 = @view V[:, 1:r]'           # V' が Vt
    v = Diagonal(S2) * Vt2         # Python: np.dot(np.diag(S), Vt)
    return U2, v, r
end

function main()
    cutoff = 1e-10

    # Bell state
    println("Bell state:")
    v = [1.0, 0.0, 0.0, 1.0] ./ sqrt(2)
    V = reshape(v, 2, 2)             # (2,2)
    F = svd(V; full=false)           # V = U * Diagonal(S) * V'
    println("singular values: ", F.S)
    U, vR, r = truncate_svd(F.U, F.S, F.Vt'; cutoff)
    println("tensors ", Any[U, vR], ")\n")

    # GHZ state
    n = 6
    println("n=$n GHZ state:")
    v = zeros(Float64, 2^n)
    v[1] = 1 / sqrt(2)           # Julia 1-based
    v[end] = 1 / sqrt(2)

    mps = Any[]
    rank = 1
    V = copy(v)

    for i in 0:(n-2)
        V = reshape(V, rank * 2, :)    # (rank*2, -1)
        F = svd(V; full=false)
        println("$(i): singular values: ", F.S)
        U2, vR, rank_new = truncate_svd(F.U, F.S, F.Vt'; cutoff)
        if i > 0
            U2 = reshape(U2, rank, 2, rank_new)
        end
        push!(mps, U2)
        V = vR
        rank = rank_new
    end
    V = reshape(V, rank, 2)
    push!(mps, V)
    println("tensors: ", mps, "\n")
end

main()
