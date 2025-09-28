#!/usr/bin/env julia
"""
QTT representation of finite-difference operator (Julia)
"""

using LinearAlgebra
using TensorOperations
using Plots

function main()
    depth = 4
    npoints = 2^depth
    cutoff = 1e-10
    max_rank = 4

    # target function and its derivative
    x = range(0.0, 1.0; length=npoints) |> collect
    y = exp.(x)
    dy = exp.(x)
    # y = cos.(x); dy = -sin.(x)
    # y = @. exp(-(((x - 0.5)/0.1)^2)) / (0.1 * sqrt(pi)); dy = @. -2.0*(x-0.5)*y/(0.1^2)

    # QTT decomposition of target function
    yt = copy(y)
    qtt = Array{Float64}[]
    rank = 1
    for k in 0:(depth-2)
        yt = reshape(permutedims(reshape(yt, rank, :, 2), (3, 1, 2)), 2 * rank, :)
        F = svd(yt; full=false)
        S = F.S
        rank_new = findall(>(cutoff * S[1]), S)
        rank_new = min(length(rank_new), max_rank)
        U = F.U[:, 1:rank_new]
        Vt = F.Vt[1:rank_new, :]
        if k > 0
            U = permutedims(reshape(U, 2, rank, rank_new), (2, 1, 3))
        end
        push!(qtt, U)
        yt = Diagonal(S[1:rank_new]) * Vt
        rank = rank_new
    end
    push!(qtt, yt)
    println("QTT virtual dimensions: ", [size(qtt[k]) for k in 1:depth], "\n")

    # QTT for index shift operator
    s_qtt = Array{Float64}[]
    s = zeros(2, 2, 2)
    s[1, 1, 1] = 1.0
    s[2, 2, 1] = 1.0
    s[2, 1, 2] = 1.0
    s[1, 2, 2] = 1.0
    push!(s_qtt, s)
    for _ in 2:(depth-1)
        s = zeros(2, 2, 2, 2)
        s[1, 1, 1, 1] = 1.0
        s[1, 2, 2, 1] = 1.0
        s[1, 2, 1, 2] = 1.0
        s[2, 1, 2, 2] = 1.0
        push!(s_qtt, s)
    end
    s = zeros(2, 2, 2)
    s[1, 2, 1] = 1.0
    s[2, 1, 2] = 1.0
    push!(s_qtt, s)
    println("index shift operator virtual dimensions: ", [size(s_qtt[k]) for k in 1:depth], "\n")

    # check for index shift operator
    if depth < 5
        sop = s_qtt[1]
        for k in 2:(depth-1)
            # not [i,l,j,m,n] but [l,i,m,j,n] because of column-major order
            @tensor tmp[l, i, m, j, n] := sop[i, j, k_] * s_qtt[k][k_, l, m, n]
            sop = reshape(tmp, size(tmp, 2) * size(tmp, 1), size(tmp, 4) * size(tmp, 3), size(tmp, 5))
        end
        @tensor tmp[l, i, m, j] := sop[i, j, k] * s_qtt[depth][k, l, m]
        sop = reshape(tmp, size(tmp, 2) * size(tmp, 1), size(tmp, 4) * size(tmp, 3))
        println("index shift operator:\n", sop, "\n")
    end

    # QTT for finite-difference operator
    d_qtt = Array{Float64}[]
    s = copy(s_qtt[1])
    st = permutedims(s, (2, 1, 3))
    d = zeros(2, 2, 2 * size(s, 3))
    @views d[:, :, 1:size(s, 3)] .= -s # minus sign for derivative
    @views d[:, :, size(s, 3)+1:end] .= st
    push!(d_qtt, d)
    for k in 2:(depth-1)
        s = copy(s_qtt[k])
        st = permutedims(s, (1, 3, 2, 4))
        d = zeros(2 * size(s, 1), 2, 2, 2 * size(s, 4))
        @views d[1:size(s, 1), :, :, 1:size(s, 4)] .= 2 .* s
        @views d[size(s, 1)+1:end, :, :, size(s, 4)+1:end] .= 2 .* st
        push!(d_qtt, d)
    end
    s = copy(s_qtt[end])
    st = permutedims(s, (1, 3, 2))
    d = zeros(2 * size(s, 1), 2, 2)
    @views d[1:size(s, 1), :, :] .= 2 .* s
    @views d[size(s, 1)+1:end, :, :] .= 2 .* st
    push!(d_qtt, d)
    println("finite-difference operator virtual dimensions: ",
        [size(d_qtt[k]) for k in 1:depth], "\n")

    # check for finite-difference operator
    if depth < 5
        dop = d_qtt[1]
        for k in 2:(depth-1)
            # not [i,l,j,m,n] but [l,i,m,j,n] because of column-major order
            @tensor tmp[l, i, m, j, n] := dop[i, j, k_] * d_qtt[k][k_, l, m, n]
            dop = reshape(tmp, size(tmp, 2) * size(tmp, 1), size(tmp, 4) * size(tmp, 3), size(tmp, 5))
        end
        @tensor tmp[l, i, m, j] := dop[i, j, k] * d_qtt[depth][k, l, m]
        dop = reshape(tmp, size(tmp, 2) * size(tmp, 1), size(tmp, 4) * size(tmp, 3))
        println("finite-difference operator:\n", dop, "\n")
    end

    # apply finite-difference operator to target function
    dy_qtt = Array{Float64}[]
    D = d_qtt[1]
    Y = qtt[1]
    @tensor tmp[j, l, n] := D[j, k, l] * Y[k, n]
    push!(dy_qtt, reshape(tmp, size(tmp, 1), size(tmp, 3) * size(tmp, 2)))
    for k in 2:(depth-1)
        D = d_qtt[k]
        Y = qtt[k]
        # not [m,i,k,n,l] but [i,m,k,l,n] because of column-major order
        @tensor tmp[i, m, j, l, n] := D[i, j, k, l] * Y[m, k, n]
        push!(dy_qtt, reshape(tmp, size(tmp, 2) * size(tmp, 1), size(tmp, 3), size(tmp, 5) * size(tmp, 4)))
    end
    D = d_qtt[end]
    Y = qtt[end]
    @tensor tmp[i, m, j] := D[i, j, k] * Y[m, k]
    push!(dy_qtt, reshape(tmp, size(tmp, 2) * size(tmp, 1), size(tmp, 3)))
    println("derivative QTT virtual dimensions: ", [size(dy_qtt[k]) for k in 1:depth], "\n")

    # reconstruction
    dyr = dy_qtt[1]
    for k in 2:(depth-1)
        # not [a,c,d] but [c,a,d] because of column-major order
        @tensor tmp[c, a, d] := dyr[a, b] * dy_qtt[k][b, c, d]
        dyr = reshape(tmp, size(tmp, 2) * size(tmp, 1), size(tmp, 3))
    end
    @tensor tmp[a, c] := dyr[a, b] * dy_qtt[end][b, c]
    dyr = vec(transpose(tmp))

    # drop boundary points as derivative is not correct there due to boundary conditions
    x2 = x[2:end-1]
    dy2 = dy[2:end-1]
    dyr2 = dyr[2:end-1]
    if depth < 5
        println("target derivative y' = ", dy2, "\n")
        println("QTT derivative   y' = ", dyr2, "\n")
    end
    err = norm(dy2 .- dyr2) / length(dy2)
    println("error = ", err, "\n")

    p1 = plot(x2, dy2, label="target")
    plot!(p1, x2, dyr2, label="QTT")
    display(p1)
    readline()

    p2 = plot(x2, dy2 .- dyr2, label="error", title="error")
    display(p2)
    readline()
end

main()
