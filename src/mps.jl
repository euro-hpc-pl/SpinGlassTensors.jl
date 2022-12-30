export
    Site,
    Sites,
    AbstractTensorNetwork,
    local_dims,
    IdentityQMps,
    move_to_CUDA!,
    device,
    MpoTensor, QMpo, QMps, TensorMap

abstract type AbstractTensorNetwork end

const Site = Union{Int, Rational{Int}}
const Sites = NTuple{N, Site} where N
const TensorMap{T} = Dict{Site, Union{Tensor{T, 2}, Tensor{T, 3}, Tensor{T, 4}}}  # 2 and 4 - mpo;  3 - mps


mutable struct MpoTensor{T <: Real, N}
    top::Vector{Tensor{T, 2}}  # N == 2 top = []
    ctr:: Union{Tensor{T, N}, Nothing}
    bot::Vector{Tensor{T, 2}}  # N == 2 bot = []
    dims::Dims{N}
end

move_to_CUDA!(ten::Nothing) = ten
device(ten::Nothing) = Set()

function move_to_CUDA!(ten::MpoTensor)
    for i in 1:length(ten.top)
        ten.top[i] = move_to_CUDA!(ten.top[i])
    end
    for i in 1:length(ten.bot)
        ten.bot[i] = move_to_CUDA!(ten.bot[i])
    end
    ten.ctr = move_to_CUDA!(ten.ctr)
    ten
end

function device(ten::MpoTensor)
    union(device(ten.ctr), device.(ten.top)..., device.(ten.bot)...)
end

Base.eltype(ten::MpoTensor{T, N}) where {T <: Real, N} = T

function MpoTensor(ten::TensorMap{T}) where T
    sk = sort(collect(keys(ten)))
    top = [ten[k] for k ∈ sk if k < 0]
    bot = [ten[k] for k ∈ sk if k > 0]
    ctr = get(ten, 0, nothing)

    if isnothing(ctr)
        top_bot = vcat(top, bot)
        dims = (0, size(top_bot[1], 1), 0, size(top_bot[end], 2))
        nn = 4
    else
        nn = ndims(ctr)
        if nn == 2
            @assert isempty(top) && isempty(bot) "Both top and bot should be empty"
            dims = size(ctr)
        elseif nn == 4
            dims = (
                size(ctr, 1), isempty(top) ? size(ctr, 2) : size(top[1], 1),
                size(ctr, 3), isempty(bot) ? size(ctr, 4) : size(bot[end], 2)
            )
        else
            throw(DomainError(ndims(ctr), "MpoTensor should have ndims 2 or 4"))
        end
    end
    MpoTensor{T, nn}(top, ctr, bot, dims)
end

#TODO should we mv this some place else?
contract_tensor3_matrix(B::AbstractArray{T, 3}, M::MpoTensor{T, 2}) where T <: Real = contract_tensor3_matrix(B, M.ctr)
contract_matrix_tensor3(M::MpoTensor{T, 2}, B::AbstractArray{T, 3}) where T <: Real = contract_matrix_tensor3(M.ctr, B)
contract_tensors43(B::Nothing, A::AbstractArray{T, 3}) where T <: Real = A

Base.ndims(ten::MpoTensor{T, N}) where {T, N} = N
Base.size(ten::MpoTensor, n::Int) = ten.dims[n]
Base.size(ten::MpoTensor) = ten.dims

const MpoTensorMap{T} = Dict{Site, MpoTensor{T}}  # MpoMap

#TODO: meta to capture both QMpo and QMps?
struct QMpo{T <: Real} <: AbstractTensorNetwork
    tensors::MpoTensorMap{T}
    sites::Vector{Site}

    function QMpo(ten::MpoTensorMap{T}) where T
        new{T}(ten, sort(collect(keys(ten))))
    end
end

struct QMps{T <: Real} <: AbstractTensorNetwork
    tensors::TensorMap{T}
    sites::Vector{Site}

    function QMps(ten::TensorMap{T}) where T
        new{T}(ten, sort(collect(keys(ten))))
    end
end

function move_to_CUDA!(ψ::Union{QMpo{T}, QMps{T}}) where T
    for k ∈ keys(ψ.tensors)
        move_to_CUDA!(ψ.tensors[k])
    end
    ψ
end

device(ψ::Union{QMpo{T}, QMps{T}}) where T = union(device(v) for v ∈ values(ψ.tensors))

@inline Base.getindex(a::AbstractTensorNetwork, i) = getindex(a.tensors, i)
@inline Base.setindex!(ket::AbstractTensorNetwork, A::AbstractArray, i::Site) = ket.tensors[i] = A
@inline Base.eltype(ψ::Union{QMpo{T}, QMps{T}}) where T = T

Base.transpose(mpo::QMpo{T}) where T <: Real = QMpo(
    MpoTensorMap{T}(keys(mpo.tensors) .=> mpo_transpose.(values(mpo.tensors)))
)

mpo_transpose(M::MpoTensor{T, 2}) where T <: Real = M

function mpo_transpose(M::MpoTensor{T, 4}) where T <: Real
    MpoTensor{T, 4}(
        mpo_transpose.(reverse(M.bot)),
        mpo_transpose(M.ctr),
        mpo_transpose.(reverse(M.top)),
        M.dims[[1, 4, 3, 2]]
    )
end

function IdentityQMps(::Type{T}, loc_dims::Dict, Dmax::Int=1) where T <: Real
    id = TensorMap{T}(keys(loc_dims) .=> CUDA.zeros.(T, Dmax, values(loc_dims), Dmax))

    site_min, ld_min = minimum(loc_dims)
    site_max, ld_max = maximum(loc_dims)
    if site_min == site_max
        id[site_min] = CUDA.zeros(T, 1, ld_min, 1)
    else
        id[site_min] = CUDA.zeros(T, 1, ld_min, Dmax)
        id[site_max] = CUDA.zeros(T, Dmax, ld_max, 1)
    end

    for (site, ld) ∈ loc_dims
        id[site][1, :, 1] .= CUDA.one(T) / sqrt(ld)
    end
    QMps(id)
end

function local_dims(mpo::QMpo, dir::Symbol)
    @assert dir ∈ (:down, :up)
    dim = dir == :down ? 4 : 2
    Dict{Site, Int}(k => size(mpo[k], dim) for k ∈ mpo.sites if ndims(mpo[k]) == 4)
end
