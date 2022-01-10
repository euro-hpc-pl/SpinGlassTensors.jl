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

function local_dims(mpo::QMpo, dir::Symbol)
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
    QMps(id)
end

@inline Base.size(tens::AbstractSparseTensor) = maximum.(tens.projs)
@inline function Base.setindex!(
    ket::AbstractTensorNetwork, A::AbstractArray, i::Site
)
    ket.tensors[i] = A
end

MPS(ket::QMps) = MPS([ket[i] for i ∈ 1:length(ket)])
QMps(ϕ::AbstractMPS) = QMps(Dict(i => A for (i, A) ∈ enumerate(ϕ)))
QMpo(ϕ::AbstractMPO) = QMpo(Dict(i => Dict(0 => A) for (i, A) ∈ enumerate(ϕ)))
