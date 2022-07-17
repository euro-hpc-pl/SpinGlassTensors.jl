export add_project, cuIdentity, cuProject

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

function __identity!(L::T, num::S) where {T <: CuDeviceMatrix, S <: Number}
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i ∈ idx:stride:size(L, 1) @inbounds L[i, i] = num end
end

"""
$(TYPEDSIGNATURES)
"""
function cuIdentity(::Type{T}, n::Int, th::Int=256) where T
    ret = CUDA.zeros(T, n, n)
    @cuda threads=th blocks=cld(n, th) __identity!(ret, one(T))
    ret
end

function __identity_permute!(L::T, pr, num::S) where {T <: CuDeviceMatrix, S <: Number}
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i ∈ idx:stride:size(L, 1) @inbounds L[pr[i], i] = num end
end

function cuProject(::Type{T}, pr, th::Int=256) where T
    n = maximum(pr)
    ret = CUDA.zeros(T, n, n)
    @cuda threads=th blocks=cld(n, th) __identity_permute!(ret, CuArray(pr), one(T))
    ret
end
