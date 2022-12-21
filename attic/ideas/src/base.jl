for T ∈ (:Site, :Diagonal, :Central, :Virtual)
    @eval begin
        ST = Symbol(:Sparse, $T, :Tensor)
        export $ST
        Base.eltype(x::$ST{R}) where R = R
    end
end

abstract type AbstractSparseTensor{T <: Real} end
abstract type AbstractTensotNetwork{T <: Real} end

const SparseTensor{T} = Union{
    SparseSiteTensor{T}, SparseDiagonalTensor{T}, SparseVirtualTensor{T}, SparseCentralTensor{T}
}
const DenseOrSparseTensor{T} = Union{SparseTensor{T}, Array{T, N}}
const Site = Union{Int, Rational{Int}}
const TensorMap = Dict{Site, DenseOrSparseTensor}
const NestedTensorMap = Dict{Site, TensorMap}
const Proj = Vector{Vector{Int}}
const MatOrCentral{T} = Union{Matrix{T}, SparseCentralTensor{T}}
const ArrayOrCuArray{T, N} = Union{Array{T, N}, CuArray{T, N}}

ArrayOrCuArray(L) = typeof(L) <: CuArray ? CuArray : Array

Base.size(M::SparseTensor, n::Int) = M.sizes[n]
Base.size(M::SparseTensor) = M.size

issparse(ten::DenseOrSparseTensor) = typeof(ten) <: AbstractSparseTensor

struct SparseSiteTensor{T <: Real} <: AbstractSparseTensor
    size::Dims
    loc_exp::Vector{T}
    projs::Proj

    function SparseSiteTensor(size, loc_exp, projs)
        new{eltype(loc_exp)}(size, loc_exp, projs)
    end
end

struct SparseDiagonalTensor{T <: Real} <: AbstractSparseTensor
    size::Dims
    e1::Matrix{T}
    e2::Matrix{T}

    function SparseDiagonalTensor(e1, e2)
        new{promote_type(eltype.((e1, e2))...)}(size, e1, e2)
    end
end

struct SparseVirtualTensor{T <: Real} <: AbstractSparseTensor
    size::Dims
    con::MatOrCentral{T}
    projs::Proj

    function SparseVirtualTensor(con, projs)
        new{eltype(con)}(size, con, projs)
    end
end

struct SparseCentralTensor{T <:Real} <: AbstractSparseTensor
    size::Dims
    vec_en::Vector{Matrix{T}}

    function SparseCentralTensor(vec_en)
        new{promote_type(eltype.(vec_en)...)}(size, vec_en)
    end
end

function Base.Array(ten::SparseCentralTensor)
    A, B, C, D = ten.vec_en
    @cast V[(u1, u2), (d1, d2)] := A[u1, d1] * B[u2, d1] * C[u1, d2] * D[u2, d2]
    V ./= maximum(V)
end

CUDA.CuArray(ten::SparseCentralTensor) = CuArray(Array(ten))

function Base.zeros(A::SparseSiteTensor{T}, B::Array{T, 3}) where T <: Real
    sal, _, sar = B.size
    sbl, _, sbt, sbr = maximum.(A.projs[1:4])
    zeros(T, sal, sbl, sbr, sar, sbt)
end

function Base.zeros(A::Array{T, 3}, B::SparseSiteTensor{T}) where T <: Real
    sal, _, sar = A.size
    sbl, sbt, sbr = maximum.(B.projs[1:3])
    zeros(T, sal, sbl, sbt, sar, sbr)
end

function Base.zero(A::SparseVirtualTensor{T}, B::Array{T, 3}) where T <: Real
    sal, _, sar = B.size
    p_lb, p_l, _, p_rb, p_r, _ = A.projs
    zeros(T, sal, length(p_l), maximum(p_lb), maximum(p_rb), sar, length(p_r))
end

function zeros(A::Array{T, 3}, B::SparseVirtualTensor{T}) where T <: Real
    sal, _, sar = A.size
    _, p_l, p_lt, _, p_r, p_rt = B.projs
    zeros(sal, length(p_l), maximum(p_lt), maximum(p_rt), sar, length(p_r))
end
