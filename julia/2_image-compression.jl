#!/usr/bin/env julia
"""
Compress and reconstruct grayscale images using SVD (Julia)
"""

using Images, ImageIO
using LinearAlgebra
using Plots

function main()
    path = "../data/sqai-square-gray-rgb150ppi.jpg"  # 入力画像パス

    # 画像を読み込み → グレイスケール化 → 行列（Float64）へ
    img = load(path)::Matrix{RGB{N0f8}}
    gimg = Gray.(img)  # RGBでもGrayに
    A01 = channelview(gimg)
    # Python版は 0..255 の画素値: スケールを合わせる
    A = A01 .* 255.0

    h, w = size(A)
    println("image size; $h $w\n")

    # 元画像の表示
    plt = heatmap(A; c=:greys, clim=(0, 255), aspect_ratio=:equal, yflip=true,
        axis=nothing, title="original image")
    display(plt)

    # SVD (thin)
    F = svd(A; full=false)   # U: H×min(H,W), S: Vector, V: W×W
    U, S, V = F.U, F.S, F.V

    # 特異値のプロット（対数軸）
    plot(S; yscale=:log10, xlabel="index", ylabel="λ_i", title="singular values")

    # ランクごとの再構成と表示
    ranks = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    for r in ranks
        rr = min(r, length(S))
        Ar = @view(U[:, 1:rr]) * Diagonal(S[1:rr]) * transpose(@view(V[:, 1:rr]))

        plt = heatmap(Ar; c=:greys, clim=(0, 255), aspect_ratio=:equal, yflip=true,
            axis=nothing, title="reconstructed image (rank $(rr))")
        display(plt)
    end
end

main()
