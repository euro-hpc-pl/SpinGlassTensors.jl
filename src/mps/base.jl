# ./mps/base.jl: This file provides basic definitions of custom Matrix Product States / Operators.

export
    Site,
    Sites,
    AbstractTensorNetwork,
    MpoTensor

abstract type AbstractTensorNetwork{T} end

const Site = Union{Int, Rational{Int}}
const Sites = NTuple{N, Site} where N
const TensorMap{T} = Dict{Site, Union{Tensor{T, 2}, Tensor{T, 3}, Tensor{T, 4}}}  # 2 and 4 - mpo;  3 - mps

mutable struct MpoTensor{T <: Real, N}
    top::Vector{Tensor{T, 2}}  # N == 2 top = []
    ctr::Union{Tensor{T, N}, Nothing}
    bot::Vector{Tensor{T, 2}}  # N == 2 bot = []
    dims::Dims{N}
end

function MpoTensor(ten::TensorMap{T}) where T
    sk = sort(collect(keys(ten)))
    top = [ten[k] for k ∈ sk if k < 0]
    bot = [ten[k] for k ∈ sk if k > 0]
    ctr = get(ten, 0, nothing)

    if isnothing(ctr)
        top_bot = vcat(top, bot)
        dims = (0, size(top_bot[1], 1), 0, size(top_bot[end], 2))
        nn = 4
    else
        nn = ndims(ctr)
        if nn == 2
            @assert isempty(top) && isempty(bot) "Both top and bot should be empty"
            dims = size(ctr)
        elseif nn == 4
            dims = (
                size(ctr, 1), isempty(top) ? size(ctr, 2) : size(top[1], 1),
                size(ctr, 3), isempty(bot) ? size(ctr, 4) : size(bot[end], 2)
            )
        else
            throw(DomainError(ndims(ctr), "MpoTensor should have ndims 2 or 4"))
        end
    end
    MpoTensor{T, nn}(top, ctr, bot, dims)
end

Base.eltype(ten::MpoTensor{T, N}) where {T, N} = T
Base.ndims(ten::MpoTensor{T, N}) where {T, N} = N
Base.size(ten::MpoTensor, n::Int) = ten.dims[n]
Base.size(ten::MpoTensor) = ten.dims

const MpoTensorMap{T} = Dict{Site, MpoTensor{T}}

for (S, M) ∈ ((:QMpo, :MpoTensorMap), (:QMps, :TensorMap))
    @eval begin
        export $S, $M
        mutable struct $S{F <: Real} <: AbstractTensorNetwork{F}
            tensors::$M{F}
            sites::Vector{Site}
            onGPU::Bool

            function $S(ten::$M{F}; onGPU::Bool=false) where F
                new{F}(ten, sort(collect(keys(ten))), onGPU)
            end
        end
    end
end

const QMpsOrMpo{T} = Union{QMpo{T}, QMps{T}}

@inline Base.getindex(ψ::QMpsOrMpo, i) = getindex(ψ.tensors, i)
@inline Base.setindex!(ψ::QMpsOrMpo, A, i::Site) = ψ.tensors[i] = A
@inline Base.eltype(ψ::QMpsOrMpo{T}) where T = T
