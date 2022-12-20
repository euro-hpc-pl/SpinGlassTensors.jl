export
    Site,
    Sites,
    AbstractTensorNetwork,
    local_dims,
    IdentityQMps,
    MpoTensor

abstract type AbstractTensorNetwork end

const Site = Union{Int, Rational{Int}}
const Sites = NTuple{N, Site} where N

const MpsTensorMap{T} = Dict{Site, Tensor{T}}

const MpoTensor{T} = Dict{Site, Tensor{T}}
const MpoTensorMap{T} = Dict{Site, MpoTensor{T}}

Base.ndims(ten::MpoTensor) = maximum(ndims.(values(ten)))

function Base.size(ten::MpoTensor)
    fk = minimum(keys(ten))
    lk = maximum(keys(ten))
    if ndims(ten) == 2
        return (size(ten[fk], 1), size(ten[lk], 2))
    elseif ndims(ten) == 4
        ddims = Dict(k => ndims(v) for (k, v) in ten)
        k, _ = maximum(ddims)
        v = ten[k]
        return (size(v, 1),
                size(ten[fk], k == fk ? 2 : 1),
                size(v, 3),
                size(ten[lk], k == lk ? 4 : 2)
                )
    else
        throw(DomainError(ndims(ten), "MpoTensor should have ndims 2 or 4"))
    end
end

maximum


Base.size(ten::MpoTensor, n::Int) = size(ten)[n]

for (F, M) ∈ ((:QMps, :MpsTensorMap), (:QMpo, :MpoTensorMap))
    @eval begin
        export $F, $M
        struct $F{T <: Real} <: AbstractTensorNetwork
            tensors::$M{T}
            sites::Vector{Site}

            function $F(ten::$M{T}) where T
                new{T}(ten, sort(collect(keys(ten))))
            end
        end
    end
end

@inline Base.getindex(a::AbstractTensorNetwork, i) = getindex(a.tensors, i)
@inline Base.setindex!(ket::AbstractTensorNetwork, A::AbstractArray, i::Site) = ket.tensors[i] = A

Base.transpose(mpo::QMpo{T}) where T <: Real = QMpo(
    MpoTensorMap{T}(keys(mpo.tensors) .=> mpo_transpose.(values(mpo.tensors)))
)

function mpo_transpose(ten::MpoTensor{T}) where T <: Real
    all(length.(size.(values(ten))) .<= 2) && return ten
    MpoTensor{T}(.- keys(ten) .=> mpo_transpose.(values(ten)))
end

function IdentityQMps(::Type{T}, loc_dims::Dict, Dmax::Int=1) where T <: Real
    id = MpsTensorMap{T}(keys(loc_dims) .=> zeros.(T, Dmax, values(loc_dims), Dmax))

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
    QMps(
        MpsTensorMap{T}(
            1 => rand(T, 1, d, D),
            (i => rand(T, D, d, D) for i ∈ 2:sites-1)...,
            sites[end] => rand(T, D, d, 1)
        )
    )
end

function Base.rand(
    ::Type{QMpo{T}}, sites::Vector, D::Int, d::Int, sites_aux::Vector=[], d_aux::Int=0
) where T <:Real
    QMpo(
        MpoTensorMap{T}(
            1 => MpoTensor{T}(
                    1 => rand(T, 1, d, d, D),
                    (j => rand(T, d_aux, d_aux) for j ∈ sites_aux)...
            ),
            sites[end] => MpoTensor{T}(
                    sites[end] => rand(T, D, d, d, 1),
                    (j => rand(T, d_aux, d_aux) for j ∈ sites_aux)...,
            ),
            (i => MpoTensor{T}(
                    i => rand(T, D, d, d, D),
                    (j => rand(T, d_aux, d_aux) for j ∈ sites_aux)...) for i ∈ 2:sites-1)...
        )
    )
end

function local_dims(mpo::QMpo, dir::Symbol)
    @assert dir ∈ (:down, :up)
    dim = dir == :down ? 4 : 2
    Dict{Site, Int}(k => size(mpo[k], dim) for k ∈ mpo.sites if ndims(mpo[k]) == 4)
end
