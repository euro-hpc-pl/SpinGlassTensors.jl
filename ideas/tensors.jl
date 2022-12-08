export
    Tensor,
    CuTensor,
    SparseTensor

for T ∈ (:SparseTensor, :Tensor, :TensorType)
    @eval begin
        AT = Symbol(:Abstract, $T)
        abstract type $AT end
        export $AT
    end
end

struct SparseTensor{T <: AbstracTensorType} <: AbstractSparseTensor
    size::Dims
    data::T

    function SparseTensor{P}(loc_exp::Vector{T}, projs::Proj) where {P <: OnSite, T <: Number}
        new{P}(size, P(loc_exp, projs))
    end

    function SparseTensor{P}(e1::Matrix{T}, e2::Matrix{T}) where {P <: Diag, T <: Number}
        new{P}(size, P(e1, e2))
    end

    function SparseTensor{Central}(vec_em) where P <: Central
        new{P}(size, P(vec_em))
    end

    function SparseTensor{P}(con::Matrix{T}, projs::Proj) where {P <: Virtual, T <: Number}
        new{P}(size, P(con, projs))
    end
end

Base.eltype(ten::SparseTensor{T}) where T = eltype(ten)

struct Tensor{T <: Number} <: AbstractTensor
    size::Dims
    data::Array{T}

    function Tensor(e11::T, e12::T, e21::T, e22::T) where T <: Matrix{<:Number}
        @cast V[(u1, u2), (d1, d2)] := e11[u1, d1] * e21[u2, d1] * e12[u1, d2] * e22[u2, d2]
        V ./= maximum(V)
        Tensor{eltype(V)}(size(V), V)
    end

    Tensor(ten::SparseTensor{T}) where T <: Central = Tensor(ten.data...)
end

struct CuTensor{T <: Number} <: AbstractTensor
    size::Dims
    data::CuArray{T}

    CuTensor(ten::SparseTensor{T}) where T <: Central = Tensor(CuArray.(ten.data)...)
end

const DenseTensor = Union{Tensor, CuTensor}
const DenseOrSparseTensor = Union{SparseTensor, Tensor}

Base.eltype(ten::DenseTensor) = eltype(ten.data)
Base.size(M::DenseOrSparseTensor, n::Int) = M.sizes[n]
Base.size(M::DenseOrSparseTensor) = M.size
