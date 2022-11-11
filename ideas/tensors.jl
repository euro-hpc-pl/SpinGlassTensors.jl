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

    function SparseTensor(loc_exp::Vector{T}, projs::Proj) where T <: Number
        new{OnSite}(size, Site(loc_exp, projs))
    end

    function SparseTensor(e1::Matrix{T}, e2::Matrix{T}) where T <: Number
        new{Diag}(size, Diag(e1, e2))
    end

    function SparseTensor(vec_em)
        new{Central}(size, Central(vec_em))
    end

    function SparseTensor(con::Matrix{T}, projs::Proj) where T <: Number
        new{Virtual}(size, Virtual(con, projs))
    end
end

Base.eltype(ten::SparseTensor{T}) where {T} = eltype(ten)

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
