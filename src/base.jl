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

const MatOrCentral{T} = Union{Matrix{T}, SparseCentralTensor{T}}

#TODO: remove AbstractArray from this
const Proj{N} = NTuple{N, AbstractArray{Int}}
const ArrayOrCuArray{N} = Union{AbstractArray{<:Real, N}, CuArray{<:Real, N}}

ArrayOrCuArray(L) = typeof(L) <: CuArray ? CuArray : Array

struct SparseSiteTensor{T <: Real} <: AbstractSparseTensor
    loc_exp::Vector{T}
    projs::Proj{N} where N

    function SparseSiteTensor(loc_exp, projs)
        T = eltype(loc_exp)
        new{T}(loc_exp, projs)
    end
end

struct SparseCentralTensor{T <: Real} <: AbstractSparseTensor
    e11::Matrix{T}
    e12::Matrix{T}
    e21::Matrix{T}
    e22::Matrix{T}
    sizes::Dims{2}

    function SparseCentralTensor(e11, e12, e21, e22, size)
        T = promote_type(eltype.((e11, e12, e21, e22))...)
        new{T}(e11, e12, e21, e22, size)
    end
end

Base.eltype(ten::SparseCentralTensor{T}) where T = T
Base.size(M::SparseCentralTensor, n::Int) = M.sizes[n]
Base.size(M::SparseCentralTensor) = M.sizes

# Base.Array(ten::SparseCentralTensor)
function dense_central_tensor(ten::SparseCentralTensor)
    @cast V[(u1, u2), (d1, d2)] := ten.e11[u1, d1] * ten.e21[u2, d1] * ten.e12[u1, d2] * ten.e22[u2, d2]
    V ./ maximum(V)
end

# CUDA.CuArray(ten::SparseCentralTensor)
function cuda_dense_central_tensor(ten::SparseCentralTensor)
    e11, e12 ,e21, e22 = CuArray.((ten.e11, ten.e12, ten.e21, ten.e22))
    @cast V[(u1, u2), (d1, d2)] := e11[u1, d1] * e21[u2, d1] * e12[u1, d2] * e22[u2, d2]
    V ./ maximum(V)
end

struct SparseDiagonalTensor{T <: Real} <: AbstractSparseTensor
    e1#::Matrix{T}
    e2#::Matrix{T}
    sizes::Dims{2}

    function SparseDiagonalTensor(e1, e2, sizes)
        T = promote_type(eltype.((e1, e2))...)
        new{T}(e1, e2, sizes)
    end
end

Base.size(M::SparseDiagonalTensor, n::Int) = M.sizes[n]
Base.size(M::SparseDiagonalTensor) = M.sizes

struct SparseVirtualTensor{T <: Real} <: AbstractSparseTensor
    con#::MatOrCentral{T}
    projs::NTuple#::Proj{N} where N

    SparseVirtualTensor(con, projs) = new{eltype(con)}(con, projs)
end

Base.size(tens::AbstractSparseTensor) = maximum.(tens.projs)

const Tensor{T} = Union{
    AbstractArray{T},
    SparseSiteTensor{T},
    SparseVirtualTensor{T},
    SparseCentralTensor{T},
    SparseDiagonalTensor{T}
}
