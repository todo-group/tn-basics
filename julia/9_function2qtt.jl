#!/usr/bin/env julia
# -*- coding: utf-8 -*-
"""
Conversion of one-dimensional function into QTT representation (Julia)
"""
using LinearAlgebra
using TensorOperations
using Plots

function main()
    depth = 8
    npoints = 2^depth
    cutoff = 1e-10
    max_rank = 4

    # target function
    x = range(0.0, 1.0; length=npoints)
    y = exp.(x)
    # y = sin.(x)
    # y = @. exp(-(((x - 0.5) / 0.1)^2)) / (0.1 * sqrt(pi))

    if depth < 5
        println("x = ", collect(x))
        println("y = ", y, "\n")
    end

    # QTT decomposition
    yt = copy(y)
    qtt = Any[]
    rank = 1
    for k in 0:(depth-2)
        println("depth: $k")
        yt_mat = reshape(yt, rank * 2, :)
        F = svd(yt_mat; full=false)
        S = F.S
        println("singular values: ", S)
        # truncate by cutoff and max_rank
        keep = findall(>(cutoff * S[1]), S)
        r = min(length(keep), max_rank)
        r == 0 && (error("All singular values truncated at depth $k"))
        U = F.U[:, 1:r]
        Vt = F.Vt[1:r, :]

        if k > 0
            U = reshape(U, rank, 2, r)
        end
        println("tensor shape: ", size(U), "\n")
        push!(qtt, U)

        yt = (Diagonal(S[1:r]) * Vt)   # next "right" core
        rank = r
    end
    yt_last = reshape(yt, rank, 2)
    println("depth: $(depth-1)")
    println("tensor shape: ", size(yt_last), "\n")
    push!(qtt, yt_last)

    if depth < 5
        println("qtt: ", qtt, "\n")
    end

    # reconstruction
    yr = qtt[1]               # (2, r1) もしくは (rank0,2,r1) だが最初は (2, r1)
    for k in 2:(depth-1)
        @tensor tmp[i, k, l] := yr[i, j] * qtt[k][j, k, l]   # (i k l)
        yr = reshape(tmp, size(tmp, 1) * size(tmp, 2), size(tmp, 3))
    end
    @tensor yr2[i, k] := yr[i, j] * qtt[end][j, k]
    # Python の reshape(-1) 相当（row-major）に合わせたい場合は行を入替後 vec
    state = vec(permutedims(yr2))  # flatten in row-major order

    err = norm(y .- state) / npoints
    println("error = $err\n")

    # plot
    plt = plot(x, y, label="target")
    if npoints <= 32
        plot!(x, state, seriestype=:scatter, label="QTT")
    else
        step = Int(cld(npoints, 32))
        plot!(x[1:step:end], state[1:step:end], seriestype=:scatter, label="QTT")
    end
    display(plt)
end

main()
