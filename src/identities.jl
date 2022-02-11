export IdentityMPO, IdentityMPS

"""
$(TYPEDEF)

$(TYPEDFIELDS)

"""
struct IdentityMPS{T <: Number, S <: AbstractArray} <: AbstractMPS{T} end

"""
$(TYPEDEF)

$(TYPEDFIELDS)

"""
struct IdentityMPO{T <: Number, S <: AbstractArray} <: AbstractMPO{T} end

"""
$(TYPEDSIGNATURES)

"""
IdentityMPS() = IdentityMPS{Float64, Array}()

"""
$(TYPEDSIGNATURES)

"""
IdentityMPO() = IdentityMPO{Float64, Array}()

"""
$(TYPEDSIGNATURES)

"""
IdentityMPS(::Type{T}) where {T <: AbstractArray} = IdentityMPS{Float64, T}

"""
$(TYPEDSIGNATURES)

"""
IdentityMPO(::Type{T}) where {T <: AbstractArray} = IdentityMPO{Float64, T}

"""
$(TYPEDSIGNATURES)

"""
IdentityMPS(::Type{S}, ::Type{T}) where {S<:Number, T<:AbstractArray} = IdentityMPS{S, T}

"""
$(TYPEDSIGNATURES)

"""
IdentityMPO(::Type{S}, ::Type{T}) where {S<:Number, T<:AbstractArray} = IdentityMPO{S, T}

const IdentityMPSorMPO = Union{IdentityMPO, IdentityMPS}

"""
$(TYPEDSIGNATURES)

"""
@inline function Base.getindex(::IdentityMPS{S, T}, ::Int) where {S, T}
    ret = similar(T{S}, (1, 1, 1))
    ret[1] = one(S)
    ret
end

"""
$(TYPEDSIGNATURES)

"""
@inline function Base.getindex(::IdentityMPO{S, T}, ::Int) where {S, T}
    ret = similar(T{S}, (1, 1, 1, 1))
    ret[1] = one(S)
    ret
end

"""
$(TYPEDSIGNATURES)

"""
LinearAlgebra.dot(O::AbstractMPO, ::IdentityMPO) = O

"""
$(TYPEDSIGNATURES)

"""
LinearAlgebra.dot(::IdentityMPO, O::AbstractMPO) = O

"""
$(TYPEDSIGNATURES)

"""
Base.length(::IdentityMPSorMPO) = Inf

"""
$(TYPEDSIGNATURES)

"""
function LinearAlgebra.dot(O::AbstractMPO, ::IdentityMPS)
    MPS([dropdims(sum(A, dims=4), dims=4) for A ∈ O])
end

"""
$(TYPEDSIGNATURES)

"""
LinearAlgebra.dot(::IdentityMPO, ψ::AbstractMPS) = ψ

"""
$(TYPEDSIGNATURES)

"""
LinearAlgebra.dot(ψ::AbstractMPS, ::IdentityMPO) = ψ

"""
$(TYPEDSIGNATURES)

"""
function Base.show(io::IO, ::IdentityMPSorMPO)
    println(io, "Trivial matrix product state")
    println(io, "   ")
end
