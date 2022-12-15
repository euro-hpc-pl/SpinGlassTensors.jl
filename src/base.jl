export
    local_dims,
    Site,
    Sites,
    State,
    Tensor,
    SparseSiteTensor,
    SparseVirtualTensor,
    SparseDiagonalTensor,
    SparseCentralTensor,
    dense_central_tensor,
    cuda_dense_central_tensor

abstract type AbstractTensorNetwork end
abstract type AbstractMPS end
abstract type AbstractMPO end
abstract type AbstractEnvironment end
abstract type AbstractSparseTensor end

const Site = Union{Int, Rational{Int}}
const Sites = NTuple{N, Site} where N
const State = Union{Vector, NTuple}

#TODO: remove AbstractArray from this union:
const ArrayOrCuArray{N} = Union{AbstractArray{<:Real, N}, CuArray{<:Real, N}}

ArrayOrCuArray(L) = typeof(L) <: CuArray ? CuArray : Array
struct SparseSiteTensor <: AbstractSparseTensor
    loc_exp::Vector{<:Real}
    projs::NTuple{N, Vector{Int}} where N
end

struct SparseCentralTensor <: AbstractSparseTensor
    e11::Matrix{<:Real}
    e12::Matrix{<:Real}
    e21::Matrix{<:Real}
    e22::Matrix{<:Real}
    sizes::Dims{2}
end

Base.size(M::SparseCentralTensor, n::Int) = M.sizes[n]
Base.size(M::SparseCentralTensor) = M.sizes

function dense_central_tensor(ten::SparseCentralTensor)
    @cast V[(u1, u2), (d1, d2)] := ten.e11[u1, d1] * ten.e21[u2, d1] * ten.e12[u1, d2] * ten.e22[u2, d2]
    V ./ maximum(V)
end

function cuda_dense_central_tensor(ten::SparseCentralTensor)
    e11, e12 ,e21, e22 = CuArray.((ten.e11, ten.e12, ten.e21, ten.e22))
    @cast V[(u1, u2), (d1, d2)] := e11[u1, d1] * e21[u2, d1] * e12[u1, d2] * e22[u2, d2]
    V ./ maximum(V)
end

struct SparseDiagonalTensor <: AbstractSparseTensor
    e1#::Matrix{<:Real}
    e2#::Matrix{<:Real}
    sizes::Dims{2}
end

Base.size(M::SparseDiagonalTensor, n::Int) = M.sizes[n]
Base.size(M::SparseDiagonalTensor) = M.sizes

struct SparseVirtualTensor <: AbstractSparseTensor
    con::Union{Matrix{<:Real}, SparseCentralTensor}
    projs::NTuple{N, Vector{Int}} where N
end

@inline Base.size(tens::AbstractSparseTensor) = maximum.(tens.projs)

const Tensor = Union{
    AbstractArray{<:Real},
    SparseSiteTensor, SparseVirtualTensor, SparseCentralTensor, SparseDiagonalTensor
}
