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

struct SiteTensor{T <: Real} <: AbstractSparseTensor
    loc_exp::Vector{T}
    projs::Proj{4}  # == pl, pt, pr, pb
    dim::Dims

    function SiteTensor(loc_exp, projs; dim=maximum.(projs))
        T = eltype(loc_exp)
        new{T}(loc_exp, projs, dim)
    end
end

function mpo_transpose(ten::SiteTensor)
    perm = [1, 4, 3, 2]
    SiteTensor(ten.loc_exp, ten.projs[perm], dim=ten.dim[perm])
end

struct CentralTensor{T <: Real} <: AbstractSparseTensor
    e11::Matrix{T}
    e12::Matrix{T}
    e21::Matrix{T}
    e22::Matrix{T}
    dim::Dims{2}

    function CentralTensor(e11, e12, e21, e22)
        s11, s12, s21, s22 = size.((e11, e12, e21, e22))
        @assert s11[1] == s12[1] && s21[1] == s22[1] && s11[2] == s21[2] && s12[2] == s22[2]
        dim = (s11[1] * s21[1], s11[2] * s12[2])
        T = promote_type(eltype.((e11, e12, e21, e22))...)
        new{T}(e11, e12, e21, e22, dim)
    end
end

mpo_transpose(ten::CentralTensor) = CentralTensor(
    transpose.((ten.e11, ten.e21, ten.e12, ten.e22))...
)

const MatOrCentral{T} = Union{Matrix{T}, CentralTensor{T}}

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

struct DiagonalTensor{T <: Real} <: AbstractSparseTensor
    e1::MatOrCentral{T}
    e2::MatOrCentral{T}
    dim::Dims{2}

    function DiagonalTensor(e1, e2)
        dim = (size(e1, 1) * size(e2, 1), size(e1, 2) * size(e2, 2))
        T = promote_type(eltype.((e1, e2))...)
        new{T}(e1, e2, dim)
    end
end

mpo_transpose(ten::DiagonalTensor) = DiagonalTensor(
    mpo_transpose.((ten.e2, ten.e1))...
)

struct VirtualTensor{T <: Real} <: AbstractSparseTensor
    con::MatOrCentral{T}
    projs::Proj{6}  # == (p_lb, p_l, p_lt, p_rb, p_r, p_rt)
    dim::Dims

    function VirtualTensor(con, projs)
        T = eltype(con)
        dim = (length(projs[2]), maximum(projs[3]) * maximum(projs[6]),
               length(projs[5]), maximum(projs[1]) * maximum(projs[4]))
        new{T}(con, projs, dim)
    end
end

mpo_transpose(ten::VirtualTensor) = VirtualTensor(ten.con, ten.projs[[3, 2, 1, 6, 5, 4]])
mpo_transpose(ten::Array{<:Real, 4}) = Array(permutedims(ten, (1, 4, 3, 2)))
mpo_transpose(ten::Array{<:Real, 2}) = Array(transpose(ten))

const SparseTensor{T} = Union{SiteTensor{T}, VirtualTensor{T}, CentralTensor{T}, DiagonalTensor{T}}
const Tensor{T} = Union{Array{T}, SparseTensor{T}}
const CentralOrDiagonal{T} = Union{CentralTensor{T}, DiagonalTensor{T}}

Base.eltype(ten::Tensor{T}) where T <: Real = T
Base.size(ten::SparseTensor, n::Int) = ten.dim[n]
Base.size(ten::SparseTensor) = ten.dim
Base.ndims(ten::SparseTensor) = length(ten.dim)
