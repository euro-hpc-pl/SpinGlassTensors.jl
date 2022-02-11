export
    QMPS,
    QMPO,
    SparseSiteTensor,
    SparseVirtualTensor,
    IdentityQMps,
    local_dims,
    Site,
    Tensor

"""
$(TYPEDEF)

$(TYPEDFIELDS)

"""
abstract type AbstractSparseTensor end

const Site = Union{Int, Rational{Int}}

"""
$(TYPEDEF)

$(TYPEDFIELDS)

"""
struct SparseSiteTensor{T} <: AbstractSparseTensor where {T <: Real}
    loc_exp::Vector{T}
    projs::NTuple{N, Vector{Int}} where N
end

"""
$(TYPEDEF)

$(TYPEDFIELDS)

"""
struct SparseVirtualTensor{T} <: AbstractSparseTensor where {T <: Real}
    con::Matrix{T}
    projs::NTuple{N, Vector{Int}} where N
end

const Tensor{T} = Union{Array{T}, SparseSiteTensor{T}, SparseVirtualTensor{T}} where {T <: Real}

"""
$(TYPEDEF)

$(TYPEDFIELDS)

"""
struct QMPS{T <: Real} <: AbstractMPS{T}
    tensors::Dict{<: Site, <: Tensor{T}}
    sites::Vector{<: Site}
end

"""
$(TYPEDSIGNATURES)

"""
function QMPS(tensors::Dict{<:Site, <:Tensor{T}}) where T
    QMPS(tensors, sort(collect(keys(tensors))))
end

"""
$(TYPEDEF)

$(TYPEDFIELDS)

"""
struct QMPO{T <: Real} <: AbstractMPO{T}
    tensors::Dict{<: Site, <: Dict{<: Site, <: Tensor{T}}}
    sites::Vector{<: Site}
end

"""
$(TYPEDSIGNATURES)

"""
function QMPO(tensors::Dict{<:Site, <:Dict{<:Site, <:Tensor{T}}}) where T
    QMPO(tensors, sort(collect(keys(tensors))))
end

"""
$(TYPEDSIGNATURES)

"""
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

"""
$(TYPEDSIGNATURES)

"""
function IdentityQMps(loc_dims::Dict, Dmax::Int=1)
    id = Dict{Site, Tensor}(site => zeros(Dmax, ld, Dmax) for (site, ld) ∈ loc_dims)
    site, ld = minimum(loc_dims)
    id[site] = zeros(1, ld, Dmax)
    site, ld = maximum(loc_dims)
    id[site] = zeros(Dmax, ld, 1)
    for (site, ld) ∈ loc_dims id[site][1, :, 1] .= 1 / sqrt(ld) end
    QMPS(id)
end

"""
$(TYPEDSIGNATURES)

"""
@inline Base.size(tens::AbstractSparseTensor) = maximum.(tens.projs)

"""
$(TYPEDSIGNATURES)

"""
@inline function Base.setindex!(
    ket::AbstractTensorNetwork, A::AbstractArray, i::Site
)
    ket.tensors[i] = A
end

"""
$(TYPEDSIGNATURES)

"""
MPS(ket::QMPS) = MPS([ket[i] for i ∈ 1:length(ket)])

"""
$(TYPEDSIGNATURES)

"""
QMPS(ϕ::AbstractMPS) = QMPS(Dict(i => A for (i, A) ∈ enumerate(ϕ)))

"""
$(TYPEDSIGNATURES)

"""
QMPO(ϕ::AbstractMPO) = QMPO(Dict(i => Dict(0 => A) for (i, A) ∈ enumerate(ϕ)))
