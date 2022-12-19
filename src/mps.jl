export
    Site,
    Sites,
    State,
    AbstractTensorNetwork,
    QMps,
    QMpo,
    local_dims,
    IdentityQMps,
    bond_dimension,
    bond_dimensions,
    NestedTensorMap,
    TensorMap


abstract type AbstractTensorNetwork end

const Site = Union{Int, Rational{Int}}
const Sites = NTuple{N, Site} where N
const State = Union{Vector, NTuple}

const TensorMap{T} = Dict{Site, Tensor{T}}
const NestedTensorMap{T} = Dict{Site, TensorMap{T}}

struct QMps{T <: Real} <: AbstractTensorNetwork
    tensors::TensorMap{T}
    sites::Vector{Site}

    function QMps(ten::TensorMap{T}) where T
        new{T}(ten, sort(collect(keys(ten))))
    end
end

struct QMpo{T <: Real} <: AbstractTensorNetwork
    tensors::NestedTensorMap{T}
    sites::Vector{Site}

    function QMpo(ten::NestedTensorMap{T}) where T
        new{T}(ten, sort(collect(keys(ten))))
    end
end

function Base.transpose(ten::TensorMap{T}) where T <: Real
    all(length.(size.(values(ten))) .<= 2) && return ten
    TensorMap{T}(.- keys(ten) .=> mpo_transpose.(values(ten)))
end

function Base.transpose(mpo::QMpo{T}) where T <: Real
    QMpo(NestedTensorMap{T}(keys(mpo.tensors) .=> transpose.(values(mpo.tensors))))
end

function IdentityQMps(::Type{T}, loc_dims::Dict, Dmax::Int=1) where T <: Real
    id = TensorMap{T}(keys(loc_dims) .=> zeros.(T, Dmax, values(loc_dims), Dmax))

    site_min, ld_min = minimum(loc_dims)
    site_max, ld_max = maximum(loc_dims)
    if site_min == site_max
        id[site_min] = zeros(T, 1, ld_min, 1)
    else
        id[site_min] = zeros(T, 1, ld_min, Dmax)
        id[site_max] = zeros(T, Dmax, ld_max, 1)
    end

    for (site, ld) ∈ loc_dims
        id[site][1, :, 1] .= one(T) / sqrt(ld)
    end
    QMps(id)
end

function Base.rand(::Type{QMps{T}}, sites::Vector, D::Int, d::Int) where T <: Real
    QMps = TensorMap{T}()
    for i ∈ sites
        if i == 1
            push!(QMps, i => randn(T, 1, d, D))
        elseif i == last(sites)
            push!(QMps, i => randn(T, D, d, 1))
        else
            push!(QMps, i => randn(T, D, d, D))
        end
    end
    QMps(QMps)
end

function Base.rand(
    ::Type{QMpo{T}}, sites::Vector, D::Int, d::Int, sites_aux::Vector=[], d_aux::Int=0
) where T <:Real
    QMpo = NestedTensorMap{T}()
    QMpo_aux = TensorMap{T}()

    for i ∈ sites
        if i == 1
            push!(QMpo_aux, i => randn(T, 1, d, d, D))
            push!(QMpo_aux, (j => randn(T, d_aux, d_aux) for j ∈ sites_aux)...)
            push!(QMpo, i => copy(QMpo_aux))
        elseif i == last(sites)
            push!(QMpo_aux, (j => randn(T, d_aux, d_aux) for j ∈ sites_aux)...)
            push!(QMpo_aux, last(sites) => randn(T, D, d, d, 1))
            push!(QMpo, i => copy(QMpo_aux))
        else
            push!(QMpo_aux, (j => randn(T, d_aux, d_aux) for j ∈ sites_aux)...)
            push!(QMpo_aux, i => randn(T, D, d, d, D))
            push!(QMpo, i => copy(QMpo_aux))
        end
        empty!(QMpo_aux)
    end
    QMpo(QMpo)
end

function local_dims(mpo::QMpo, dir::Symbol)
    @assert dir ∈ (:down, :up)
    lds = Dict{Site, Int}()
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

@inline Base.getindex(a::AbstractTensorNetwork, i) = getindex(a.tensors, i)
@inline Base.setindex!(ket::AbstractTensorNetwork, A::AbstractArray, i::Site) = ket.tensors[i] = A
