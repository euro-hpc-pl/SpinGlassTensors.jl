export
    AbstractTensorNetwork,
    AbstractMPS,
    AbstractMPO,
    QMps,
    QMpo,
    local_dims,
    IdentityQMps,
    #random_QMps,
    #random_QMpo,
    bond_dimension,
    bond_dimensions,
    verify_bonds,
    is_left_normalized,
    is_right_normalized

abstract type AbstractTensorNetwork end
abstract type AbstractMPS end
abstract type AbstractMPO end

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

function IdentityQMps(::Type{T}, loc_dims::Dict, Dmax::Int=1) where T <: Number
    id = Dict{Site, Tensor{T}}(
        site => zeros(T, Dmax, ld, Dmax) for (site, ld) ∈ loc_dims
    )
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
IdentityQMps(loc_dims::Dict, Dmax::Int=1) = IdentityQMps(Float64, loc_dims, Dmax)

@inline Base.getindex(a::AbstractTensorNetwork, i) = getindex(a.tensors, i)
@inline Base.size(a::AbstractTensorNetwork) = (length(a.tensors), )
@inline Base.length(a::AbstractTensorNetwork) = length(a.tensors)
@inline LinearAlgebra.rank(ψ::QMps) = Tuple(size(A, 2) for A ∈ values(ψ.tensors))
@inline bond_dimension(ψ::QMps) = maximum(size.(values(ψ.tensors), 3))
@inline bond_dimensions(ψ::QMps) = [size(ψ.tensors[n]) for n in ψ.sites]
@inline Base.copy(ψ::QMps) = QMps(copy(ψ.tensors))
@inline Base.:(≈)(a::QMps, b::QMps) = isapprox(a.tensors, b.tensors)
@inline Base.:(≈)(a::QMpo, b::QMpo) = all([isapprox(a.tensors[i], b.tensors[i]) for i ∈ keys(a.tensors)])
@inline Base.setindex!(ket::AbstractTensorNetwork, A::AbstractArray, i::Site) = ket.tensors[i] = A

function Base.isapprox(l::Dict, r::Dict)
    l === r && return true
    length(l) != length(r) && return false
    for pair ∈ l
        !in(pair, r, isapprox) && return false
    end
    true
end

#=
function verify_bonds(ψ::QMps)
    L = length(ψ.sites)
    @assert size(ψ.tensors[1], 1) == 1 "Incorrect size on the left boundary."
    @assert size(ψ.tensors[L], 3) == 1 "Incorrect size on the right boundary."
    for i ∈ 1:L-1
        @assert size(ψ.tensors[i], 3) == size(ψ.tensors[i+1], 1) "Incorrect link between $i and $(i+1)."
    end
end
=#

#=
function random_QMps(sites::Vector, D::Int, d::Int)
    qmps = Dict{Site, Tensor}()
    for i ∈ sites
        if i == 1
            push!(qmps, i => randn(1, d, D))
        elseif i == last(sites)
            push!(qmps, i => randn(D, d, 1))
        else
            push!(qmps, i => randn(D, d, D))
        end
    end
    QMps(qmps)
end

function random_QMpo(
    ::Type{T}, sites::Vector, D::Int, d::Int, sites_aux::Vector=[], d_aux::Int=0
) where T <: Number
    qmpo = Dict{Site, Dict{Site, Tensor}}()
    qmpo_aux = Dict{Site, Tensor}()

    for i ∈ sites
        if i == 1
            push!(qmpo_aux, i => randn(T, 1, d, d, D))
            push!(qmpo_aux, (j => randn(T, d_aux, d_aux) for j ∈ sites_aux)...)
            push!(qmpo, i => copy(qmpo_aux))
        elseif i == last(sites)
            push!(qmpo_aux, (j => randn(T, d_aux, d_aux) for j ∈ sites_aux)...)
            push!(qmpo_aux, last(sites) => randn(T, D, d, d, 1))
            push!(qmpo, i => copy(qmpo_aux))
        else
            push!(qmpo_aux, (j => randn(T, d_aux, d_aux) for j ∈ sites_aux)...)
            push!(qmpo_aux, i => randn(T, D, d, d, D))
            push!(qmpo, i => copy(qmpo_aux))
        end
        empty!(qmpo_aux)
    end
    QMpo(qmpo)
end

random_QMpo(sites::Vector, D::Int, d::Int, sites_aux::Vector=[], d_aux::Int=0) =
random_QMpo(Float64, sites, D, d, sites_aux, d_aux)
=#

function Base.rand(::Type{QMps{T}}, sites::Vector, D::Int, d::Int) where T <: Real
    QMps = Dict{Site, Array{T, 3}}()
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
) where T <: Number
    QMpo = Dict{Site, Dict{Site, Array{T, 4}}}()
    QMpo_aux = Dict{Site, Array{T, 4}}()

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

#=
function is_left_normalized(ψ::QMps)
    all(
       I(size(A, 3)) ≈ @tensor Id[x, y] := conj(A)[α, σ, x] * A[α, σ, y] order = (α, σ)
       for A ∈ values(ψ.tensors)
    )
end

function is_right_normalized(ϕ::QMps)
    all(
        I(size(B, 1)) ≈ @tensor Id[x, y] := B[x, σ, α] * conj(B)[y, σ, α] order = (α, σ)
        for B in values(ϕ.tensors)
    )
end
=#
