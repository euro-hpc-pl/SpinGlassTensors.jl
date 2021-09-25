export 
    Mps, 
    Mpo


abstract type AbstractEnvironment end
abstract type AbstractMps end
abstract type AbstractMpo end


const AbstractTN = Union{AbstractMps, AbstractMpo}
const Site = Union{Int, Rational{Int}} 

mutable struct Mps <: AbstractMps
    tensors
    sites
    Mps(ket::Dict) = new(ket, sort(collect(keys(ket))))
end


@inline Base.getindex(ket::AbstractTN, i) = getindex(ket.tensors, i)
@inline Base.setindex!(ket::AbstractTN, A::AbstractArray, i::Int) = ket.tensors[i] = A
@inline Base.length(ket::AbstractTN) = length(ket.tensors)
@inline Base.copy(ket::AbstractTN) = Mps(copy(ket.tensors))


mutable struct Mpo <: AbstractMpo
    tensors
    sites
    Mpo(op::Dict) = new(op, sort(collect(keys(op))))
end


@inline Base.copy(ket::AbstractMPO) = Mpo(copy(ket.tensors))


function MPS(ket::Mps)
    L = length(ket)
    ϕ = MPS(eltype(ket[1]), L) 
    for i ∈ 1:L ϕ[i] = ket[i] end
    ϕ
end


Mps(ϕ::AbstractMPS) = Mps(Dict(i => A for (i, A) ∈ enumerate(ϕ)))
Mpo(ϕ::AbstractMPO) = Mpo(Dict(i => A for (i, A) ∈ enumerate(ϕ)))
