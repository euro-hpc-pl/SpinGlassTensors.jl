for S ∈ (:OnSite, :Diag, :Central, :Virtual)
    @eval begin
        export $S
        Base.eltype(x::$S{T}) where T = T
    end
end

abstract type AbstractTensotNetwork{T <: Number} end

const Site = Union{Int, Rational{Int}}
const TensorMap = Dict{Site, DenseOrSparseTensor}
const NestedTensorMap = Dict{Site, TensorMap}
const Proj = Vector{Vector{Int}}

struct OnSite{T <: Number}
    loc_exp::Vector{T}
    projs::Proj

    OnSite(loc_exp, projs) = new{eltype(loc_exp)}(loc_exp, projs)
end

struct Diag{T <: Number} <: AbstractSparseTensor
    e1::Matrix{T}
    e2::Matrix{T}

    Diag(e1, e2) = new{promote_type(eltype(e1), eltype(e2))}(e1, e2)
end

struct Virtual{T <: Number} <: AbstractSparseTensor
    con::Matrix{T}
    projs::Proj
    s
    Virtual(con, projs) = new{eltype(con)}(con, projs)
end

struct Central{T <: Number} <: AbstractSparseTensor
    vec_en::Vector{Matrix{T}}

    Central(vec_en) = new{promote_type(eltype.(vec_en)...)}(vec_en)
end

function Base.eltype(ten::Union{TensorMap, NestedTensorMap})
    promote_type(eltype.(values(ten))...)
end
