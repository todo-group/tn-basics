#!/usr/bin/env julia
"""
Compress and reconstruct grayscale images using SVD
"""

using Images, ImageIO
using LinearAlgebra
using Plots

function main()
    path = "../data/sqai-square-gray-rgb150ppi.jpg"  # path to input image

    # load image and convert to grayscale
    image = load(path)::Matrix{RGB{N0f8}}
    A = channelview(Gray.(image)) .* 255.0
    h, w = size(A)
    println("image size; $h $w\n")
    plt = heatmap(A; c=:greys, clim=(0, 255), aspect_ratio=:equal, yflip=true,
        axis=nothing, title="original image")
    display(plt)
    readline()

    # SVD
    F = svd(A; full=false)
    U, S, V = F.U, F.S, F.V
    plt = plot(S; yscale=:log10, xlabel="index", ylabel="λ_i", title="singular values")
    display(plt)
    readline()

    # image reconstruction with different ranks
    ranks = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    for r in ranks
        rr = min(r, length(S))
        Ar = @view(U[:, 1:rr]) * Diagonal(S[1:rr]) * transpose(@view(V[:, 1:rr]))
        plt = heatmap(Ar; c=:greys, clim=(0, 255), aspect_ratio=:equal, yflip=true,
            axis=nothing, title="reconstructed image (rank $(rr))")
        display(plt)
        readline()
    end
end

main()
