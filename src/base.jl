export
    Tensor,
    SiteTensor,
    VirtualTensor,
    DiagonalTensor,
    CentralTensor,
    CentralOrDiagonal,
    mpo_transpose

abstract type AbstractSparseTensor end

const Proj{N} = NTuple{N, Array{Int, 1}}
const CuArrayOrArray{T, N} = Union{Array{T, N}, CuArray{T, N}}

ArrayOrCuArray(L) = typeof(L) <: CuArray ? CuArray : Array # TODO do we need this?

struct SiteTensor{T <: Real, N} <: AbstractSparseTensor
    loc_exp::Vector{T}
    projs::Proj{N}  # == pl, pt, pr, pb
    dims::Dims{N}

    function SiteTensor(loc_exp, projs; dims=maximum.(projs))
        @assert length(dims) == 4  # N = 4
        T = eltype(loc_exp)
        new{T, 4}(loc_exp, projs, dims)
    end
end

function mpo_transpose(ten::SiteTensor)
    perm = [1, 4, 3, 2]
    SiteTensor(ten.loc_exp, ten.projs[perm], dims=ten.dims[perm])
end

struct CentralTensor{T <: Real, N} <: AbstractSparseTensor
    e11::Matrix{T}
    e12::Matrix{T}
    e21::Matrix{T}
    e22::Matrix{T}
    dims::Dims{N}

    function CentralTensor(e11, e12, e21, e22)
        s11, s12, s21, s22 = size.((e11, e12, e21, e22))
        @assert s11[1] == s12[1] && s21[1] == s22[1] &&
                s11[2] == s21[2] && s12[2] == s22[2]
        dims = (s11[1] * s21[1], s11[2] * s12[2])
        T = promote_type(eltype.((e11, e12, e21, e22))...)
        new{T, 2}(e11, e12, e21, e22, dims)
    end
end

mpo_transpose(ten::CentralTensor) = CentralTensor(transpose.((ten.e11, ten.e21, ten.e12, ten.e22))...)

const MatOrCentral{T, N} = Union{Matrix{T}, CentralTensor{T, N}}

function Base.Array(ten::CentralTensor)
    @cast V[(u1, u2), (d1, d2)] := ten.e11[u1, d1] * ten.e21[u2, d1] *
                                   ten.e12[u1, d2] * ten.e22[u2, d2]
    V ./ maximum(V)
end

function CUDA.CuArray(ten::CentralTensor)
    e11, e12 ,e21, e22 = CuArray.((ten.e11, ten.e12, ten.e21, ten.e22))
    @cast V[(u1, u2), (d1, d2)] := e11[u1, d1] * e21[u2, d1] *
                                   e12[u1, d2] * e22[u2, d2]
    V ./ maximum(V)
end

struct DiagonalTensor{T <: Real, N} <: AbstractSparseTensor
    e1::MatOrCentral{T, N}
    e2::MatOrCentral{T, N}
    dims::Dims{N}

    function DiagonalTensor(e1, e2)
        dims = (size(e1, 1) * size(e2, 1), size(e1, 2) * size(e2, 2))
        T = promote_type(eltype.((e1, e2))...)
        new{T, 2}(e1, e2, dims)
    end
end

mpo_transpose(ten::DiagonalTensor) = DiagonalTensor(mpo_transpose.((ten.e2, ten.e1))...)

struct VirtualTensor{T <: Real, N} <: AbstractSparseTensor
    con::MatOrCentral{T, 2}
    projs::Proj{6}  # == (p_lb, p_l, p_lt, p_rb, p_r, p_rt)
    dims::Dims{N}

    function VirtualTensor(con, projs)
        T = eltype(con)
        dims = (length(projs[2]), maximum(projs[3]) * maximum(projs[6]),
                length(projs[5]), maximum(projs[1]) * maximum(projs[4]))
        new{T, 4}(con, projs, dims)
    end
end

mpo_transpose(ten::VirtualTensor) = VirtualTensor(ten.con, ten.projs[[3, 2, 1, 6, 5, 4]])
mpo_transpose(ten::Array{<:Real, 4}) = Array(permutedims(ten, (1, 4, 3, 2)))
mpo_transpose(ten::Array{<:Real, 2}) = Array(transpose(ten))

const SparseTensor{T, N} = Union{
    SiteTensor{T, N}, VirtualTensor{T, N}, CentralTensor{T, N}, DiagonalTensor{T, N}
}
const Tensor{T, N} = Union{Array{T, N}, SparseTensor{T, N}}
const CentralOrDiagonal{T, N} = Union{CentralTensor{T, N}, DiagonalTensor{T, N}}

Base.eltype(ten::Tensor{T, N}) where {T <: Real, N} = T
Base.ndims(ten::Tensor{T, N}) where {T, N} = N
Base.size(ten::SparseTensor, n::Int) = ten.dims[n]
Base.size(ten::SparseTensor) = ten.dims
