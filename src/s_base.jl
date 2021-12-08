export QMps, QMpo, local_dims, SparseSiteTensor, SparseVirtualTensor, IdentityQMps

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

struct QMps <: AbstractTensorNetwork{Number}
    tensors::Dict
    sites::Vector{Site}
    QMps(tensors::Dict) = new(tensors, sort(collect(keys(tensors))))
end

struct QMpo <: AbstractTensorNetwork{Number}
    tensors::Dict # of Dict
    sites::Vector{Site}
    QMpo(tensors::Dict) = new(tensors, sort(collect(keys(tensors))))
end

# local_dim(mpo::QMpo, site::Site, dir::Symbol) = local_dim(mpo, site, Val(dir))
# function local_dim(mpo::QMpo, site::Site, ::Val{:up})
#     mkeys = sort(collect(keys(mpo[site])))
#     size(mpo[site][first(mkeys)], 2)
# end

# function local_dim(mpo::QMpo, site::Site, ::Val{:down})
#     mkeys = sort(collect(keys(mpo[site])))
#     size(mpo[site][last(mkeys)], 4)
# end

function local_dims(mpo::QMpo, dir::Symbol)
    @assert dir ∈ (:down, :up)
    lds = Dict()
    for site ∈ mpo.sites
        mkeys = sort(collect(keys(mpo[site])))
        if any(length(size(mpo[site][kk])) > 2 for kk ∈ mkeys)
            if dir == :down
                ss = size(mpo[site][last(mkeys)])
                ld = length(ss) == 4 ? ss[4] : ss[2]
            else
                ss = size(mpo[site][first(mkeys)])
                ld = length(ss) == 4 ? ss[2] : ss[1]
            end
            push!(lds, site => ld)
        end
    end
    lds
end


function IdentityQMps(loc_dims, Dmax::Int=1)
    sites = sort(collect(keys(loc_dims)))
    id = Dict()
    for i ∈ 1:length(sites)
        push!(id, sites[i] => zeros(Dmax, loc_dims[sites[i]], Dmax))
    end
    id[sites[begin]] = zeros(1, loc_dims[sites[begin]], Dmax)
    id[sites[end]] = zeros(Dmax, loc_dims[sites[end]], 1)
    for i ∈ 1:length(sites)
        id[sites[i]][1, :, 1] .= 1 / sqrt(loc_dims[sites[i]])
    end
    QMps(id)
end


@inline Base.size(tens::AbstractSparseTensor) = collect(maximum(pr) for pr ∈ tens.projs)
@inline function Base.setindex!(
    ket::AbstractTensorNetwork, A::AbstractArray, i::Site
)
    ket.tensors[i] = A
end

MPS(ket::QMps) = MPS([ket[i] for i ∈ 1:length(ket)])
QMps(ϕ::AbstractMPS) = QMps(Dict(i => A for (i, A) ∈ enumerate(ϕ)))
QMpo(ϕ::AbstractMPO) = QMpo(Dict(i => Dict(0 => A) for (i, A) ∈ enumerate(ϕ)))
