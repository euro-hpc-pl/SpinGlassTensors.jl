export
    Mps, Mpo,
    local_dims,
    SparseSiteTensor,
    SparseVirtualTensor


abstract type AbstractEnvironment end
abstract type AbstractSparseTensor end

const Site = Union{Int, Rational{Int}} 

struct SparseSiteTensor <: AbstractSparseTensor
    loc_exp::Vector{Real}
    projs::NTuple{N, Vector{Int}} where N
end


struct SparseVirtualTensor <: AbstractSparseTensor
    con::Vector{Real}
    projs::NTuple{N, Vector{Int}} where N
end


struct Mps <: AbstractTensorNetwork{Number} 
    tensors::Dict
    sites::Vector{Site}
    Mps(tensors::Dict) = 
    new(tensors, sort(collect(keys(tensors))))
end


struct Mpo <: AbstractTensorNetwork{Number} 
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
@inline Base.setindex!(ket::AbstractTensorNetwork, A::AbstractArray, i::Site) = ket.tensors[i] = A

MPS(ket::Mps) = MPS([ket[i] for i ∈ 1:length(ket)])

Mps(ϕ::AbstractMPS) = Mps(Dict(i => A for (i, A) ∈ enumerate(ϕ)))
Mpo(ϕ::AbstractMPO) = Mpo(Dict(i => Dict(0 => A) for (i, A) ∈ enumerate(ϕ)))
