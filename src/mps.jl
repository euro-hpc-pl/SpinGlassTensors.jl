export
    Site,
    Sites,
    AbstractTensorNetwork,
    local_dims,
    IdentityQMps,
    MpoTensor, QMpo, QMps, TensorMap

abstract type AbstractTensorNetwork end

const Site = Union{Int, Rational{Int}}
const Sites = NTuple{N, Site} where N

const TensorMap{T} = Dict{Site, Union{Tensor{T, 2}, Tensor{T, 3}, Tensor{T, 4}}}  # 2 and 4 for MPO;  3 for mps

struct QMps{T <: Real} <: AbstractTensorNetwork
    tensors::TensorMap{T}
    sites::Vector{Site}

    function QMps(ten::TensorMap{T}) where T
        new{T}(ten, sort(collect(keys(ten))))
    end
end

struct MpoTensor{T <: Real, N}
    top::Vector{Tensor{T, 2}}  # N == 2 top = []
    ctr:: Union{Tensor{T, N}, Type{Nothing}}
    bot::Vector{Tensor{T, 2}}  # N == 2 bot = []
    dims::Dims{N}

    function MpoTensor(ten::TensorMap{T}) where T
        sk = sort(collect(keys(ten)))
        top = [ten[k] for k ∈ sk if k < 0]
        bot = [ten[k] for k ∈ sk if k > 0]
        ctr = get(ten, 0, Nothing)

        if ctr == Nothing
            top_bot = vcat(top, bot)
            dims = (size(top_bot[1], 1), size(top_bot[end], 2))
            nn = 2
        else
            nn = ndims(ctr)
            if nn == 2
                @assert length(top) == length(bot) == 0
                dims = size(ctr)
            elseif nn == 4
                dims = (size(ctr, 1),
                    length(top) == 0 ? site(ctr, 2) : size(first(top), 1),
                    size(ctr, 3),
                    length(bot) == 0 ? site(ctr, 4) : size(last(bot), 2)
                    )
            else
                throw(DomainError(ndims(ctr), "MpoTensor will have ndims 2 or 4"))
            end
        end
        new{T, nn}(top, ctr, bot, dims)
    end
end

Base.ndims(ten::MpoTensor{T, N}) where {T, N} = N
Base.size(ten::MpoTensor, n::Int) = ten.dims[n]
Base.size(ten::MpoTensor) = ten.dims



const MpoTensorMap{T} = Dict{Site, MpoTensor{T}}

struct QMpo{T <: Real} <: AbstractTensorNetwork
    tensors::MpoTensorMap{T}
    sites::Vector{Site}

    function QMpo(ten::MpoTensorMap{T}) where T
        new{T}(ten, sort(collect(keys(ten))))
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
