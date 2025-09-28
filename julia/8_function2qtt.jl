#!/usr/bin/env julia
"""
Conversion of one-dimensional function into QTT representation (Julia)
"""

using LinearAlgebra
using TensorOperations
using Plots

function main()
    depth = 4
    npoints = 2^depth
    cutoff = 1e-10
    max_rank = 4

    # target function
    x = range(0.0, 1.0; length=npoints)
    y = exp.(x)
    # y = cos.(x)
    # y = @. exp(-(((x - 0.5) / 0.1)^2)) / (0.1 * sqrt(pi))

    if depth < 5
        println("x = ", collect(x))
        println("y = ", y, "\n")
    end

    # QTT decomposition
    yt = copy(y)
    qtt = Array{Float64}[]
    rank = 1
    for k in 0:(depth-2)
        println("depth: $k")
        yt = reshape(permutedims(reshape(yt, rank, :, 2), (3, 1, 2)), 2 * rank, :)
        F = svd(yt; full=false)
        S = F.S
        println("singular values: ", S)
        rank_new = findall(>(cutoff * S[1]), S)
        rank_new = min(length(rank_new), max_rank)
        U = F.U[:, 1:rank_new]
        Vt = F.Vt[1:rank_new, :]
        if k > 0
            U = permutedims(reshape(U, 2, rank, rank_new), (2, 1, 3))
        end
        push!(qtt, U)
        yt = (Diagonal(S[1:rank_new]) * Vt)
        rank = rank_new
    end
    println("depth: $(depth-1)")
    push!(qtt, yt)
    if depth < 5
        println("qtt: ", qtt, "\n")
    end

    # reconstruction
    yr = qtt[1]::Matrix{Float64}
    for k in 2:(depth-1)
        # not [i, k, l] but [k, i, l] because of column-major order
        @tensor tmp[k, i, l] := yr[i, j] * qtt[k][j, k, l]
        yr = reshape(tmp, size(tmp, 2) * size(tmp, 1), size(tmp, 3))
    end
    @tensor yr2[i, k] := yr[i, j] * qtt[end][j, k]
    state = vec(transpose(yr2))
    err = norm(y .- state) / npoints
    println("error = $err\n")

    if depth < 5
        println("reconstructed y = ", state, "\n")
    end
    plt = plot(x, y, label="target")
    if npoints <= 32
        plot!(x, state, seriestype=:scatter, label="QTT")
    else
        step = Int(cld(npoints, 32))
        plot!(x[1:step:end], state[1:step:end], seriestype=:scatter, label="QTT")
    end
    display(plt)

    plt = plot(x, y - state, label="error")
    display(plt)
end

main()
