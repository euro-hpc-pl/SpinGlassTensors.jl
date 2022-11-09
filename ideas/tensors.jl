export
    Tensor,
    SparseTensor

for T ∈ (:SparseTensor, :Tensor, :TensorType)
    @eval begin
        AT = Symbol(:Abstract, $T)
        abstract type $AT end
        export $AT
    end
end

const Proj = Vector{Vector{Int}}

struct Site{T <: Number}
    loc_exp::Vector{T}
    projs::Proj

    function Site(loc_exp, projs)
        S = eltype(loc_exp)
        new{S}(loc_exp, projs)
    end
end

struct Diag{T <: Number} <: AbstractSparseTensor
    e1::Matrix{T}
    e2::Matrix{T}

    function Diag(e1, e2)
        S = promote_type(eltype(e1), eltype(e2))
        new{S}(e1, e2)
    end
end

struct Virtual{T <: Number} <: AbstractSparseTensor
    con::Matrix{T}
    projs::Proj

    function Virtual(con, projs)
        S = eltype(con)
        new{S}(con, projs)
    end
end

struct Central{T <: Number} <: AbstractSparseTensor
    vec_en::Vector{Matrix{T}}

    function Central(vec_en)
        S = promote_type(eltype.(vec_en)...)
        new{S}(vec_en)
    end
end

struct PegasusSquare{T <: Number} <: AbstractSparseTensor
    projs::Proj
    loc_exp::Matrix{T}
    bnd_exp::Vector{Matrix{T}}
    bnd_projs::Proj

    function PegasusSquar(proj, loc_exp, bnd_exp, bnd_projs)
        S = promote_type(eltype(loc_exp), eltype(bnd_exp))
        new{S}(proj, loc_exp, bnd_exp, bnd_projs)
    end
end

for S ∈ (:Site, :Diag, :Central, :Virtual, :PegasusSquare)
    @eval begin
        export $S
        Base.eltype(x::$S{T}) where {T} = T
    end
end

struct SparseTensor{T <: AbstracTensorType} <: AbstractSparseTensor
    size::Dims
    data::T

    function SparseTensor(loc_exp, projs)
        new{Site}(size, Site(loc_exp, projs))
    end

    function SparseTensor(e1, e2)
        new{Diag}(size, Diag(e1, e2))
    end

    function SparseTensor(vec_em)
        new{Central}(size, Central(vec_em))
    end

    function SparseTensor(con, projs)
        new{Virtual}(size, Virtual(con, projs))
    end

    function SparseTensor(proj, loc_exp, bnd_exp, bnd_projs)
        new{PegasusSquare}(size, PegasusSquare(proj, loc_exp, bnd_exp, bnd_projs))
    end
end

Base.eltype(ten::SparseTensor{T}) where {T} = eltype(ten)

struct Tensor{T <: Number} <: AbstractTensor
    size::Dims
    data::Array{T}

    function Tensor(e11::T, e12::T, e21::T, e22::T) where T <: Matrix{<:Number}
        @cast V[(u1, u2), (d1, d2)] := e11[u1, d1] * e21[u2, d1] * e12[u1, d2] * e22[u2, d2]
        W = V ./ maximum(V)
        Tensor{eltype(ten)}(size(W), W)
    end

    Tensor(ten::SparseTensor{Central}) = Tensor(ten.data...)
end

struct CuTensor{T <: Number} <: AbstractTensor
    size::Dims
    data::CuArray{T}

    CuTensor(ten::SparseTensor{Central}) = Tensor(CuArray.(ten.data)...)
end

const DenseTensor = Union{Tensor, CuTensor}
const DenseOrSparseTensor = Union{SparseTensor, Tensors}

Base.eltype(ten::DenseTensor) = eltype(ten.data)
Base.size(M::DenseOrSparseTensor, n::Int) = M.sizes[n]
Base.size(M::DenseOrSparseTensor) = M.size
