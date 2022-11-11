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

    function OnSite(loc_exp, projs)
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

function Base.eltype(ten::Union{TensorMap, NestedTensorMap})
    promote_type(eltype.(values(ten))...)
end
