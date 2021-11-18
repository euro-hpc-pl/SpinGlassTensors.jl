export
    Mps, Mpo,
    local_dims,
    SparseSiteTensor,
    SparseVirtualTensor


abstract type AbstractEnvironment end
abstract type AbstractSparseTensor end

abstract type AbstractMps end
abstract type AbstractMpo end


const AbstractTN = Union{AbstractMps, AbstractMpo}
const Site = Union{Int, Rational{Int}} 


struct SparseSiteTensor <: AbstractSparseTensor
    loc_exp::Vector{Real}
    projs::NTuple{N, Vector{Int}} where N
end


struct SparseVirtualTensor <: AbstractSparseTensor
    con::Vector{Real}
    projs::NTuple{N, Vector{Int}} where N
end


struct Mps <: AbstractMps
    tensors::Dict
    sites::Vector{Site}
    Mps(tensors::Dict) = 
    new(tensors, sort(collect(keys(tensors))))
end


struct Mpo <: AbstractMpo
    tensors::Dict # of Dict
    sites::Vector{Site}
    Mpo(tensors::Dict) = 
    new(tensors, sort(collect(keys(tensors))))
end


local_dim(mpo::Mpo, site::Site, dir::Symbol) = local_dim(mpo, site, Val(dir))


function local_dim(mpo::Mpo, site::Site, ::Val{:up})
    mkeys = sort(collect(keys(mpo[site])))
    size(mpo[site][first(mkeys)], 2)
end


function local_dim(mpo::Mpo, site::Site, ::Val{:down})
    mkeys = sort(collect(keys(mpo[site])))
    size(mpo[site][last(mkeys)], 4)
end


function local_dims(mpo::Mpo, dir::Symbol)
    @assert dir ∈ (:down, :up)
    Dict(site => local_dim(mpo, site, dir) for site ∈ mpo.sites)
end


@inline Base.size(tens::AbstractSparseTensor, ind::Int) = maximum(tens.projs[ind])
@inline Base.getindex(ket::AbstractTN, i) = getindex(ket.tensors, i)
@inline Base.setindex!(ket::AbstractTN, A::AbstractArray, i::Site) = ket.tensors[i] = A
@inline Base.length(ket::AbstractTN) = length(ket.tensors)
@inline Base.iterate(a::AbstractTN) = iterate(a.tensors)
@inline Base.iterate(a::AbstractTN, state) = iterate(a.tensors, state)


function MPS(ket::Mps)
    L = length(ket)
    ϕ = MPS(eltype(ket[1]), L) 
    for i ∈ 1:L ϕ[i] = ket[i] end
    ϕ
end


Mps(ϕ::AbstractMPS) = Mps(Dict(i => A for (i, A) ∈ enumerate(ϕ)))
Mpo(ϕ::AbstractMPO) = Mpo(Dict(i => Dict(0 => A) for (i, A) ∈ enumerate(ϕ)))



