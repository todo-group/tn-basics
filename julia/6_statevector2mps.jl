#!/usr/bin/env julia
"""
Generate MPS from statevector
"""

using LinearAlgebra

function main()
    cutoff = 1e-10

    # Bell state
    println("Bell state:")
    v = [1.0, 0.0, 0.0, 1.0] ./ sqrt(2)

    v = permutedims(reshape(v, 2, 2))
    F = svd(v; full=false)
    println("singular values: ", F.S)
    rank_new = sum(F.S .> cutoff * F.S[1])
    U = @view F.U[:, 1:rank_new]
    S = Diagonal(F.S[1:rank_new])
    Vt = @view(F.V[:, 1:rank_new])'
    v = S * Vt
    println("tensors ", Any[U, v], ")\n")

    # GHZ state
    n = 16
    println("n=$n GHZ state:")
    v = zeros(Float64, 2^n)
    v[1] = 1 / sqrt(2)
    v[end] = 1 / sqrt(2)

    mps = Array{Float64}[]
    rank = 1
    for i in 0:(n-2)
        v = reshape(v, rank * 2, :)
        F = svd(v; full=false)
        println("$(i): singular values: ", F.S)
        rank_new = sum(F.S .> cutoff * F.S[1])
        U = @view F.U[:, 1:rank_new]
        S = Diagonal(F.S[1:rank_new])
        Vt = @view(F.V[:, 1:rank_new])'
        if i > 0
            U = reshape(U, rank, 2, rank_new)
        end
        push!(mps, U)
        v = S * Vt
        rank = rank_new
    end
    v = reshape(v, rank, 2)
    push!(mps, v)
    println("tensors: ", mps, "\n")

    # random state
    n = 16
    println("n=$n random state:")
    v = rand(2^n)
    v = v / norm(v)

    mps = Array{Float64}[]
    rank = 1
    for i in 0:(n-2)
        v = reshape(v, rank * 2, :)
        F = svd(v; full=false)
        println("$(i): singular values: ", F.S)
        rank_new = sum(F.S .> cutoff * F.S[1])
        U = @view F.U[:, 1:rank_new]
        S = Diagonal(F.S[1:rank_new])
        Vt = @view(F.V[:, 1:rank_new])'
        if i > 0
            U = reshape(U, rank, 2, rank_new)
        end
        push!(mps, U)
        v = S * Vt
        rank = rank_new
    end
    v = reshape(v, rank, 2)
    push!(mps, v)
    println("virtual bond dimensions:")
    for i in 2:n
        print("$(i-1): $(size(mps[i])[1])\n")
    end
end

main()
