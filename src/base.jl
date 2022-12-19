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

ArrayOrCuArray(L) = typeof(L) <: CuArray ? CuArray : Array

struct SiteTensor{T <: Real} <: AbstractSparseTensor
    loc_exp::Vector{T}
    projs::Proj{4}  # TODO p1, p2, p3, p4 ?
    size::Dims

    function SiteTensor(loc_exp, projs; size=maximum.(projs))
        T = eltype(loc_exp)
        new{T}(loc_exp, projs, size)
    end
end

function mpo_transpose(ten::SiteTensor)
    perm = [1, 4, 3, 2]
    SiteTensor(ten.loc_exp, ten.projs[perm], size=ten.size[perm])
end
struct CentralTensor{T <: Real} <: AbstractSparseTensor
    e11::Matrix{T}
    e12::Matrix{T}
    e21::Matrix{T}
    e22::Matrix{T}
    size::Dims{2}

    function CentralTensor(e11, e12, e21, e22, size)
        T = promote_type(eltype.((e11, e12, e21, e22))...)
        new{T}(e11, e12, e21, e22, size)
    end
end

function mpo_transpose(ten::CentralTensor)
    CentralTensor(transpose.((ten.e11, ten.e21, ten.e12, ten.e22))..., ten.size[[2, 1]])
end

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
    size::Dims{2}

    function DiagonalTensor(e1, e2, size)
        T = promote_type(eltype.((e1, e2))...)
        new{T}(e1, e2, size)
    end
end

function mpo_transpose(ten::DiagonalTensor)
    DiagonalTensor(mpo_transpose(ten.e2), mpo_transpose(ten.e1), ten.size[[2, 1]])
end

struct VirtualTensor{T <: Real} <: AbstractSparseTensor
    con::MatOrCentral{T}
    projs::Proj{6}  # == (p_lb, p_l, p_lt, p_rb, p_r, p_rt)
    size::Dims  # TODO should be the size of virtual tensor; not size of virtual legs

    function VirtualTensor(con, projs)
        T = eltype(con)
        new{T}(con, projs, maximum.(projs))
    end
end

function mpo_transpose(ten::VirtualTensor)
    VirtualTensor(ten.con, ten.projs[[3, 2, 1, 6, 5, 4]])
end

mpo_transpose(ten::Array{T, 4}) where T = Array(permutedims(ten, (1, 4, 3, 2)))
mpo_transpose(ten::Array{T, 2}) where T = Array(transpose(ten))


function Base.zeros(A::SiteTensor{T}, B::Array{T, 3}) where T <: Real
    sal, _, sar = size(B)
    sbl, _, sbt, sbr = maximum.(A.projs[1:4])
    zeros(T, sal, sbl, sbr, sar, sbt)
end

function Base.zeros(A::Array{T, 3}, B::SiteTensor{T}) where T <: Real
    sal, _, sar = size(A)
    sbl, sbt, sbr = maximum.(B.projs[1:3])
    zeros(T, sal, sbl, sbt, sar, sbr)
end

function Base.zero(A::VirtualTensor{T}, B::Array{T, 3}) where T <: Real
    sal, _, sar = size(B)
    p_lb, p_l, _, p_rb, p_r, _ = A.projs
    zeros(T, sal, length(p_l), maximum(p_lb), maximum(p_rb), sar, length(p_r))
end

function Base.zeros(A::Array{T, 3}, B::VirtualTensor{T}) where T <: Real
    sal, _, sar = size(A)
    _, p_l, p_lt, _, p_r, p_rt = B.projs
    zeros(sal, length(p_l), maximum(p_lt), maximum(p_rt), sar, length(p_r))
end

const SparseTensor{T} = Union{SiteTensor{T}, VirtualTensor{T}, CentralTensor{T}, DiagonalTensor{T}}
const Tensor{T} = Union{Array{T}, SparseTensor{T}}
const CentralOrDiagonal{T} = Union{CentralTensor{T}, DiagonalTensor{T}}

Base.eltype(ten::Tensor{T}) where T = T
Base.size(ten::SparseTensor, n::Int) = ten.size[n]
Base.size(ten::SparseTensor) = ten.size
