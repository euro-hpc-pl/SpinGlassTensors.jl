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
    loc_exp
    projs
end


struct SparseVirtualTensor <: AbstractSparseTensor
    con
    projs
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
    Mpo(tensors::Dict)  = 
    new(tensors, sort(collect(keys(tensors))))
end


function local_dims(mpo::Mpo, dir::Symbol)
    @assert dir ∈ (:down, :up)
    dims = Dict{Site, Int}()
    if dir == :up
        for site ∈ mpo.sites
            mkeys = sort(collect(keys(mpo[site])))
            T = mpo[site][first(mkeys)]
            push!(dims, site => size(T, 2))
        end
    else
        for site ∈ mpo.sites
            mkeys = sort(collect(keys(mpo[site])))
            T = mpo[site][last(mkeys)]
            push!(dims, site => size(T, 4))
        end
    end
    dims
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



