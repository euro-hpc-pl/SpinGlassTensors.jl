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
@inline Base.copy(ket::AbstractTN) = AbstractTN(copy(ket.tensors))

mutable struct Mpo <: AbstractMpo
    tensors
    sites
    Mpo(op::Dict) = new(op, sort(collect(keys(op))))
end

