#!/usr/bin/env julia
# -*- coding: utf-8 -*-
"""
(Simplified) TEBD simulation of random quantum circuits — Julia
"""

using LinearAlgebra
using Random
using TensorOperations

const C64 = ComplexF64

"単一量子ビットのランダムU(α,θ,φ)"
function random_u()::Matrix{C64}
    α   = 2π * rand()
    θ   = π  * rand()
    ϕ   = 2π * rand()
    eα  = cis(α)
    e₊ϕ = cis(ϕ)
    e₋ϕ = cis(-ϕ)
    return eα * [
        cos(θ/2)            -e₊ϕ*sin(θ/2)
        e₋ϕ*sin(θ/2)         cos(θ/2)
    ]
end

"state(2,2,...,2) へ pos の1量子ビットUを適用（全状態テンソル版）"
function apply_1q!(state::Array{C64}, pos::Int, U::Matrix{C64})
    n = ndims(state)
    order = vcat(pos, setdiff(1:n, pos))
    invorder = invperm(order)
    st = permutedims(state, order)                # (2, 2, ..., 2)
    st2 = reshape(st, 2, :)
    st2 = U * st2                                  # 左から作用
    st = reshape(st2, size(st))
    state .= permutedims(st, invorder)
    return state
end

"state(2,2,...,2) へ pos,pos+1 の2量子ビットG(2,2,2,2) を適用"
function apply_2q!(state::Array{C64}, pos::Int, G::Array{C64,4})
    n = ndims(state)
    order = vcat(pos, pos+1, setdiff(1:n, [pos, pos+1]))
    invorder = invperm(order)
    st = permutedims(state, order)                 # (2,2,2,2, ..., 2)
    st2 = reshape(st, 2, 2, :)
    @tensor st3[a,b,t] := G[a,b,c,d] * st2[c,d,t] # 先頭2脚へゲート
    st = reshape(st3, size(st))
    state .= permutedims(st, invorder)
    return state
end

"2サイトMPS更新: mps[i], mps[i+1] と G(2,2,2,2) からSVD分割（cutoff/χmax）"
function mps_two_site_update!(mps::Vector{Array{C64,3}}, i::Int;
                              G::Array{C64,4}, cutoff=1e-10, χmax=typemax(Int))
    A = mps[i]         # (χL, 2, r)
    B = mps[i+1]       # (r,  2, χR)

    # 2サイトにゲートを作用
    @tensor T[χl,i1,i2,χr] := G[i1,i2,j1,j2] * A[χl,j1,α] * B[α,j2,χr]
    Tmat = reshape(T, size(T,1)*size(T,2), size(T,3)*size(T,4))

    U,S,Vt = svd(Tmat; full=false)
    # しきい値/最大ボンド次元でトリム
    keep = findall(x -> x > cutoff, S)
    r = min(length(keep), χmax)
    r == 0 && (return (mps[i] .= zeros(C64, size(A,1),2,1);
                       mps[i+1] .= zeros(C64, 1,2,size(B,3))))
    S = S[1:r]
    U = U[:, 1:r]
    Vt = Vt[1:r, :]

    Ssqrt = Diagonal(sqrt.(S))
    Anew = U * Ssqrt
    Bnew = Ssqrt * Vt

    mps[i]   = reshape(Anew, size(A,1), 2, r)
    mps[i+1] = reshape(Bnew, r, 2, size(B,3))
end

function main()
    n       = 16      # qubits
    depth   = 16      # layers
    max_dim = 4       # χmax for truncated MPS
    cutoff  = 1e-10

    # CNOT (control = first, target = second)
    cnot = Array{C64}(undef, 2,2,2,2)
    cnot .= 0
    cnot[1,1,1,1] = 1
    cnot[1,2,1,2] = 1
    cnot[2,1,2,2] = 1
    cnot[2,2,2,1] = 1

    # |0...0> statevector (tensor形: (2,2,...,2))
    state = zeros(C64, ntuple(_->2, n))
    state[ntuple(_->1, n)...] = 1

    # MPS 初期化（|0⟩ = [1,0]^T）
    mps0 = [reshape([C64(1),C64(0)], 1,2,1) for _ in 1:n]
    mps1 = [copy(x) for x in mps0]
    println("mps0/1 initial virtual bond dimensions: ",
            [size(mps0[i],3) for i in 1:n-1])

    for k in 0:depth-1
        # ランダム1量子ゲート
        for pos in 1:n
            U = random_u()
            apply_1q!(state, pos, U)
            @tensor mps0[pos][a,i,b] := U[i,j] * mps0[pos][a,j,b]
            @tensor mps1[pos][a,i,b] := U[i,j] * mps1[pos][a,j,b]
        end

        # CNOT レイヤ（オッド/イーブン交互）
        for i in (k÷2 + 1):2:(n-1)
            apply_2q!(state, i, cnot)
            # 無切断
            mps_two_site_update!(mps0, i; G=cnot, cutoff=cutoff, χmax=typemax(Int))
            # 切断あり
            mps_two_site_update!(mps1, i; G=cnot, cutoff=cutoff, χmax=max_dim)
        end

        # MPS から状態再構成（mps0）
        ψ0 = mps0[1]
        ψ0 = reshape(ψ0, 2, size(ψ0,3))
        for i in 2:n
            @tensor tmp[a,c,d] := ψ0[a,b] * mps0[i][b,c,d]
            ψ0 = reshape(tmp, size(tmp,1)*size(tmp,2), size(tmp,3))
        end
        ψ0_vec = vec(permutedims(ψ0))  # row-major flatten

        # MPS から状態再構成（mps1）
        ψ1 = mps1[1]
        ψ1 = reshape(ψ1, 2, size(ψ1,3))
        for i in 2:n
            @tensor tmp[a,c,d] := ψ1[a,b] * mps1[i][b,c,d]
            ψ1 = reshape(tmp, size(tmp,1)*size(tmp,2), size(tmp,3))
        end
        ψ1_vec = vec(permutedims(ψ1))

        # フル state をベクトル化（row-major）
        st_mat = reshape(state, 2, :)
        st_vec = vec(permutedims(st_mat))

        fid0 = abs(dot(conj.(st_vec), ψ0_vec))^2
        fid1 = abs(dot(conj.(st_vec), ψ1_vec))^2
        println("step: $k fidelity: $fid0, $fid1")
    end

    println("mps0 final virtual bond dimensions: ", [size(mps0[i],3) for i in 1:n-1])
    println("mps1 final virtual bond dimensions: ", [size(mps1[i],3) for i in 1:n-1])
end

main()
