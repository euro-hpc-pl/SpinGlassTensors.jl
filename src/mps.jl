export
    Site,
    Sites,
    AbstractTensorNetwork,
    local_dims,
    IdentityQMps

abstract type AbstractTensorNetwork end

const Site = Union{Int, Rational{Int}}
const Sites = NTuple{N, Site} where N

const TensorMap{T} = Dict{Site, Tensor{T}}
const NestedTensorMap{T} = Dict{Site, TensorMap{T}}

for (F, M) ∈ ((:QMps, :TensorMap), (:QMpo, :NestedTensorMap))
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
    NestedTensorMap{T}(keys(mpo.tensors) .=> mpo_transpose.(values(mpo.tensors)))
)

function mpo_transpose(ten::TensorMap{T}) where T <: Real
    all(length.(size.(values(ten))) .<= 2) && return ten
    TensorMap{T}(.- keys(ten) .=> mpo_transpose.(values(ten)))
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
    QMps(
        TensorMap{T}(
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
        NestedTensorMap{T}(
            1 => TensorMap{T}(
                    1 => rand(T, 1, d, d, D),
                    (j => rand(T, d_aux, d_aux) for j ∈ sites_aux)...
            ),
            sites[end] => TensorMap{T}(
                    sites[end] => rand(T, D, d, d, 1),
                    (j => rand(T, d_aux, d_aux) for j ∈ sites_aux)...,
            ),
            (i => TensorMap{T}(
                    i => rand(T, D, d, d, D),
                    (j => rand(T, d_aux, d_aux) for j ∈ sites_aux)...) for i ∈ 2:sites-1)...
        )
    )
end

# TODO rm this?
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
