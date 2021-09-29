export 
    Mps, 
    Mpo,
    is_left_normalized,
    is_right_normalized


abstract type AbstractEnvironment end
abstract type AbstractMps end
abstract type AbstractMpo end


const AbstractTN = Union{AbstractMps, AbstractMpo}
const Site = Union{Int, Rational{Int}} 

mutable struct Mps <: AbstractMps
    tensors
    sites
    Mps(tensors::Dict) = new(tensors, sort(collect(keys(tensors))))
end


@inline Base.getindex(ket::AbstractTN, i) = getindex(ket.tensors, i)
@inline Base.setindex!(ket::AbstractTN, A::AbstractArray, i::Int) = ket.tensors[i] = A
@inline Base.length(ket::AbstractTN) = length(ket.tensors)
@inline Base.copy(ket::AbstractTN) = AbstractTN(copy(ket.tensors))


mutable struct Mpo <: AbstractMpo
    tensors
    sites
    Mpo(tensors::Dict) = new(tensors, sort(collect(keys(tensors))))
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


function is_left_normalized(ψ::Mps)
    all(
        I(size(A, 3)) ≈ @tensor Id[x, y] := conj(A[α, σ, x]) * A[α, σ, y] order = (α, σ) for A ∈ ψ
        )
end


function is_right_normalized(ϕ::Mps)
    all(
    I(size(B, 1)) ≈ @tensor Id[x, y] := B[x, σ, α] * conj(B[y, σ, α]) order = (α, σ) for B ∈ ϕ
    )
end
