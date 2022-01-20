export 
    QMPS,
    QMPO,
    SparseSiteTensor,
    SparseVirtualTensor,
    IdentityQMps,
    local_dims

abstract type AbstractEnvironment end
abstract type AbstractSparseTensor end

const Site = Union{Int, Rational{Int}}

struct SparseSiteTensor <: AbstractSparseTensor
    loc_exp::Vector{<:Real}
    projs::NTuple{N, Vector{Int}} where N
end

struct SparseVirtualTensor <: AbstractSparseTensor
    con::Matrix{<:Real}
    projs::NTuple{N, Vector{Int}} where N
end

struct QMPS{T <: Number} <: AbstractMPS{T}
    tensors::Dict{Site, Array{T, 3}}
    sites::Vector{Site}
end

function QMPS(tensors::Dict{<:Site, Array{T, 3}}) where {T <: Number}
    QMPS{T}(tensors, sort(collect(keys(tensors))))
end

struct QMPO{T <: Number} <: AbstractMPO{T}
    tensors::Dict{Site, Dict{Site, Array{T, 4}}}
    sites::Vector{Site}
end

function QMPO(tensors::Dict{<:Site, <:Dict{<:Site, Array{T, 4}}}) where {T <: Number}
    QMPO{T}(tensors, sort(collect(keys(tensors))))
end

function local_dims(mpo::QMPO, dir::Symbol)
    @assert dir ∈ (:down, :up)
    lds = Dict()
    for site ∈ mpo.sites
        mkeys = sort(collect(keys(mpo[site])))
        if any(length(size(mpo[site][k])) > 2 for k ∈ mkeys)
            if dir == :down
                ss = size(mpo[site][last(mkeys)])
                push!(lds, site => length(ss) == 4 ? ss[4] : ss[2])
            else
                ss = size(mpo[site][first(mkeys)])
                push!(lds, site => length(ss) == 4 ? ss[2] : ss[1])
            end
        end
    end
    lds
end

function IdentityQMps(loc_dims::Dict, Dmax::Int=1)
    id = Dict(site => zeros(Dmax, ld, Dmax) for (site, ld) ∈ loc_dims)
    site, ld = minimum(loc_dims)
    id[site] = zeros(1, ld, Dmax)
    site, ld = maximum(loc_dims)
    id[site] = zeros(Dmax, ld, 1)
    for (site, ld) ∈ loc_dims id[site][1, :, 1] .= 1 / sqrt(ld) end
    QMPS(id)
end

@inline Base.size(tens::AbstractSparseTensor) = maximum.(tens.projs)
@inline function Base.setindex!(
    ket::AbstractTensorNetwork, A::AbstractArray, i::Site
)
    ket.tensors[i] = A
end

MPS(ket::QMPS) = MPS([ket[i] for i ∈ 1:length(ket)])
QMPS(ϕ::AbstractMPS) = QMPS(Dict(i => A for (i, A) ∈ enumerate(ϕ)))
QMPO(ϕ::AbstractMPO) = QMPO(Dict(i => Dict(0 => A) for (i, A) ∈ enumerate(ϕ)))
