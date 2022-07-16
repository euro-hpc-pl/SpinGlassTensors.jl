export add_project

function __add_project!(ret, L, pr)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    x_stride = gridDim().x * blockDim().x

    idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    y_stride = gridDim().y * blockDim().y

    for i ∈ idx:x_stride:size(L, 1), j ∈ idy:y_stride:size(L, 2)
        for (σ, p) ∈ enumerate(pr)
            @inbounds ret[i, j, p] += L[i, j, σ]
        end
    end
end

function add_project(L, pr, th=(16, 16))
    n, m, _ = size(L)
    bl = cld.((n, m), th)

    proj = CUDA.CuArray(pr)
    ret = CUDA.zeros(eltype(L), n, m, maximum(pr))

    CUDA.@sync @cuda threads=th blocks=bl __add_project!(ret, L, proj)
    ret
end
