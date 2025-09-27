#!/usr/bin/env julia
# -*- coding: utf-8 -*-
"""
QTT representation of finite-difference operator (Julia)
"""
using LinearAlgebra
using TensorOperations
using Plots

function main()
    depth   = 4
    npoints = 2^depth
    cutoff  = 1e-10
    max_rank = 4

    # target function & derivative
    x  = range(0.0, 1.0; length=npoints) |> collect
    y  = exp.(x)
    dy = exp.(x)
    # y  = cos.(x);  dy = -sin.(x)
    # y  = @. exp(-(((x - 0.5)/0.1)^2)) / (0.1 * sqrt(pi));  dy = @. -2.0*(x-0.5)*y/(0.1^2)

    # QTT decomposition of y
    yt   = copy(y)
    qtt  = Vector{Array{Float64,3}}()
    rank = 1
    for k in 0:(depth-2)
        yt_mat = reshape(yt, rank*2, :)
        F = svd(yt_mat; full=false)
        S = F.S
        keep = findall(>(cutoff*S[1]), S)
        r = min(length(keep), max_rank)
        U = F.U[:, 1:r]
        Vt = F.Vt[1:r, :]
        push!(qtt, reshape(U, rank, 2, r))
        yt = Diagonal(S[1:r]) * Vt
        rank = r
    end
    push!(qtt, reshape(yt, rank, 2, 1))
    println("QTT virtual dimensions: ", [size(qtt[k]) for k in 1:depth], "\n")

    # QTT for index shift operator (left shift; sameビット構成)
    s_qtt = Vector{Array{Float64,4}}()
    s = zeros(1,2,2,2)
    s[1,1,1,1] = 1.0; s[1,2,2,1] = 1.0; s[1,2,1,2] = 1.0; s[1,1,2,2] = 1.0
    push!(s_qtt, s)
    for _ in 2:(depth-1)
        s = zeros(2,2,2,2)
        s[1,1,1,1] = 1.0; s[1,2,2,1] = 1.0; s[1,2,1,2] = 1.0; s[2,1,2,2] = 1.0
        push!(s_qtt, s)
    end
    s = zeros(2,2,2,1)
    s[1,2,1,1] = 1.0; s[2,1,2,1] = 1.0
    push!(s_qtt, s)
    println("index shift operator virtual dimensions: ", [size(s_qtt[k]) for k in 1:depth], "\n")

    # check (only for small depth)
    if depth < 5
        # build dense shift matrix of size npoints×npoints
        sop = reshape(s_qtt[1], 2,2,2)
        for k in 2:depth
            @tensor tmp[i,l,j,m,n] := sop[i,j,k_] * s_qtt[k][k_,l,m,n]
            sop = reshape(tmp, size(tmp,1)*size(tmp,2), size(tmp,3)*size(tmp,4), size(tmp,5))
        end
        sop = reshape(sop, npoints, npoints)
        println("index shift operator:\n", sop, "\n")
    end

    # QTT for finite-difference operator (D ≈ (Shift - Shift^T)/Δx with scaling factors)
    d_qtt = Vector{Array{Float64,4}}()
    # first core
    s  = copy(s_qtt[1]);  st = permutedims(s, (1,3,2,4))
    d  = zeros(size(s,1), 2,2, 2*size(s,4))
    @views d[:,:,:, 1:size(s,4)]      .= s
    @views d[:,:,:, size(s,4)+1:end]  .= -st
    push!(d_qtt, d)
    # middle cores
    for k in 2:(depth-1)
        s  = copy(s_qtt[k]);  st = permutedims(s, (1,3,2,4))
        d  = zeros(2*size(s,1), 2,2, 2*size(s,4))
        @views d[1:size(s,1),:,:, 1:size(s,4)]               .= 2 .* s
        @views d[size(s,1)+1:end,:,:, size(s,4)+1:end]        .= 2 .* st
        push!(d_qtt, d)
    end
    # last core
    s  = copy(s_qtt[end]);  st = permutedims(s, (1,3,2,4))
    d  = zeros(2*size(s,1), 2,2, size(s,4))
    @views d[1:size(s,1),:,:,:]      .= 2 .* s
    @views d[size(s,1)+1:end,:,:,:]  .= 2 .* st
    push!(d_qtt, d)
    println("finite-difference operator virtual dimensions: ",
            [size(d_qtt[k]) for k in 1:depth], "\n")

    if depth < 5
        dop = reshape(d_qtt[1], 2,2,4)
        for k in 2:depth
            @tensor tmp[i,l,j,m,n] := dop[i,j,k_] * d_qtt[k][k_,l,m,n]
            dop = reshape(tmp, size(tmp,1)*size(tmp,2), size(tmp,3)*size(tmp,4), size(tmp,5))
        end
        dop = reshape(dop, npoints, npoints)
        println("finite-difference operator:\n", dop, "\n")
    end

    # apply finite-difference operator to QTT(y)
    dy_qtt = Vector{Array{Float64,3}}()
    for k in 1:depth
        D = d_qtt[k];  Y = qtt[k]
        # t = einsum("ijkl,mjn->miknl", D, Y) → reshape (mi, k, nl)
        @tensor tmp[m,i,k,n,l] := D[i,j,k,l] * Y[m,j,n]
        push!(dy_qtt, reshape(tmp, size(tmp,1)*size(tmp,2), size(tmp,3), size(tmp,4)*size(tmp,5)))
    end
    println("derivative QTT virtual dimensions: ", [size(dy_qtt[k]) for k in 1:depth], "\n")

    # reconstruction
    dyr = dy_qtt[1] |> x->reshape(x, size(x,1)*size(x,2), size(x,3))
    for k in 2:depth
        @tensor tmp[a,c,d] := dyr[a,b] * dy_qtt[k][b,c,d]
        dyr = reshape(tmp, size(tmp,1)*size(tmp,2), size(tmp,3))
    end
    # flatten (row-major)
    dyr = vec(permutedims(dyr))

    # drop boundary points (not accurate at edges)
    x2  = x[2:end-1]
    dy2 = dy[2:end-1]
    dyr2 = dyr[2:end-1]
    println("target derivative y' = ", dy2, "\n")
    println("QTT derivative   y' = ", dyr2, "\n")
    err = norm(dy2 .- dyr2) / length(dy2)
    println("error = ", err, "\n")

    # plots
    p1 = plot(x2, dy2, label="target")
    plot!(p1, x2, dyr2, label="QTT 1")
    display(p1)

    p2 = plot(x2, dy2 .- dyr2, label="error 1", title="error")
    display(p2)
end

main()
