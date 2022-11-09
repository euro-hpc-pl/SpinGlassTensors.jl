export
    AbstractTensorNetwork,
    AbstractMPS,
    AbstractMPO,
    QMps,
    QMpo,
    local_dims,
    Site,
    Sites,
    State,
    Tensor,
    SparseSiteTensor,
    SparseVirtualTensor,
    SparseDiagonalTensor,
    SparseCentralTensor,
    dense_central_tensor,
    cuda_dense_central_tensor,
    IdentityQMps,
    random_QMps,
    random_QMpo,
    bond_dimension,
    verify_bonds,
    is_left_normalized,
    is_right_normalized

abstract type AbstractTensorNetwork{T} end
abstract type AbstractMPS end
abstract type AbstractMPO end
abstract type AbstractEnvironment end
abstract type AbstractSparseTensor end

const Site = Union{Int, Rational{Int}}
const Sites = NTuple{N, Site} where N
const State = Union{Vector, NTuple}



struct SparseSiteTensor <: AbstractSparseTensor
    loc_exp::Vector{<:Real}
    projs::NTuple{N, Vector{Int}} where N
end


struct SparseCentralTensor <: AbstractSparseTensor
    e11::Matrix{<:Real}
    e12::Matrix{<:Real}
    e21::Matrix{<:Real}
    e22::Matrix{<:Real}
    sizes::NTuple{2, Int}
end

Base.size(M::SparseCentralTensor, n::Int) = M.sizes[n]
Base.size(M::SparseCentralTensor) = M.sizes

function dense_central_tensor(ten::SparseCentralTensor)
    @cast V[(u1, u2), (d1, d2)] := ten.e11[u1, d1] * ten.e21[u2, d1] * ten.e12[u1, d2] * ten.e22[u2, d2]
    V ./ maximum(V)
end

function cuda_dense_central_tensor(ten::SparseCentralTensor)
    e11 = CUDA.CuArray(ten.e11)
    e12 = CUDA.CuArray(ten.e12)
    e21 = CUDA.CuArray(ten.e21)
    e22 = CUDA.CuArray(ten.e22)
    @cast V[(u1, u2), (d1, d2)] := e11[u1, d1] * e21[u2, d1] * e12[u1, d2] * e22[u2, d2]
    V ./ maximum(V)
end


struct SparseDiagonalTensor <: AbstractSparseTensor
    e1 #::Matrix{<:Real}
    e2 #::Matrix{<:Real}
    sizes::NTuple{2, Int}
end

Base.size(M::SparseDiagonalTensor, n::Int) = M.sizes[n]
Base.size(M::SparseDiagonalTensor) = M.sizes


#TODO: potentially change name. Used in SquareStar geometry.
"""
$(TYPEDSIGNATURES)
"""
struct SparseVirtualTensor <: AbstractSparseTensor
    con::Union{Matrix{<:Real}, SparseCentralTensor}
    projs::NTuple{N, Vector{Int}} where N
end

"""
$(TYPEDSIGNATURES)
"""
@inline Base.size(tens::AbstractSparseTensor) = maximum.(tens.projs)


"""
$(TYPEDSIGNATURES)
"""
@inline Base.getindex(a::AbstractTensorNetwork, i) = getindex(a.tensors, i)


"""
$(TYPEDSIGNATURES)
"""
const Tensor = Union{AbstractArray{Float64}, SparseSiteTensor, SparseVirtualTensor, SparseCentralTensor, SparseDiagonalTensor}


#TODO: type of sites
"""
$(TYPEDSIGNATURES)
"""
struct QMps <: AbstractTensorNetwork{Number}
    tensors::Dict{Site, Tensor}
    sites::Vector{Site}
    QMps(tensors::Dict{<:Site, <:Tensor}) = new(tensors, sort(collect(keys(tensors))))
end

"""
$(TYPEDSIGNATURES)
"""
struct QMpo <: AbstractTensorNetwork{Number}
    tensors::Dict{Site, Dict{Site, Tensor}}
    sites::Vector{Site}
    function QMpo(tensors::Dict{<:Site, <:Dict{<:Site, <:Tensor}})
        new(tensors, sort(collect(keys(tensors))))
    end
end

#TODO: rethink this function
"""
$(TYPEDSIGNATURES)
"""
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

"""
$(TYPEDSIGNATURES)
"""
function IdentityQMps(loc_dims::Dict, Dmax::Int=1)
    id = Dict{Site, Tensor}(site => zeros(Dmax, ld, Dmax) for (site, ld) ∈ loc_dims)
    site_min, ld_min = minimum(loc_dims)
    site_max, ld_max = maximum(loc_dims)
    if site_min == site_max
        id[site_min] = zeros(1, ld_min, 1)
    else
        id[site_min] = zeros(1, ld_min, Dmax)
        id[site_max] = zeros(Dmax, ld_max, 1)
    end
        for (site, ld) ∈ loc_dims id[site][1, :, 1] .= 1 / sqrt(ld) end
    QMps(id)
end


"""
$(TYPEDSIGNATURES)
"""
@inline Base.size(a::AbstractTensorNetwork) = (length(a.tensors), )

"""
$(TYPEDSIGNATURES)
"""
@inline Base.length(a::AbstractTensorNetwork) = length(a.tensors)

"""
$(TYPEDSIGNATURES)
"""
@inline LinearAlgebra.rank(ψ::QMps) = Tuple(size(A, 2) for A ∈ values(ψ.tensors))

"""
$(TYPEDSIGNATURES)
"""
@inline bond_dimension(ψ::QMps) = maximum(size.(values(ψ.tensors), 3))

"""
$(TYPEDSIGNATURES)
"""
Base.copy(ψ::QMps) = QMps(copy(ψ.tensors))

"""
$(TYPEDSIGNATURES)
"""
@inline Base.:(≈)(a::QMps, b::QMps) = isapprox(a.tensors, b.tensors)

"""
$(TYPEDSIGNATURES)
"""
@inline Base.:(≈)(a::QMpo, b::QMpo) = all([isapprox(a.tensors[i], b.tensors[i]) for i in keys(a.tensors)])


"""
$(TYPEDSIGNATURES)
"""
function Base.isapprox(l::Dict, r::Dict)
    l === r && return true
    if length(l) != length(r) return false end
    for pair in l
        if !in(pair, r, isapprox)
            return false
        end
    end
    true
end

"""
$(TYPEDSIGNATURES)
"""
function verify_bonds(ψ::QMps)
    L = length(ψ.sites)

    @assert size(ψ.tensors[1], 1) == 1 "Incorrect size on the left boundary."
    @assert size(ψ.tensors[L], 3) == 1 "Incorrect size on the right boundary."

    for i ∈ 1:L-1
        @assert size(ψ.tensors[i], 3) == size(ψ.tensors[i+1], 1) "Incorrect link between $i and $(i+1)."
    end
end

"""
$(TYPEDSIGNATURES)
"""
@inline function Base.setindex!(
    ket::AbstractTensorNetwork, A::AbstractArray, i::Site
)
    ket.tensors[i] = A
end

function random_QMps(sites::Vector, D::Int, d::Int)
    qmps = Dict{Site, Tensor}()
    for i in sites
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

function random_QMpo(sites::Vector, D::Int, d::Int, sites_aux::Vector=[], d_aux::Int=0)
    qmpo = Dict{Site, Dict{Site, Tensor}}()
    qmpo_aux = Dict{Site, Tensor}()

    for i in sites
        if i == 1
            push!(qmpo_aux, i => randn(1, d, d, D))
            [push!(qmpo_aux, j => randn(d_aux, d_aux)) for j in sites_aux]
            new_qmpo_aux = copy(qmpo_aux)
            push!(qmpo, i => new_qmpo_aux)
        elseif i == last(sites)
            for j in sites_aux
                push!(qmpo_aux, j => randn(d_aux, d_aux))
            end
            push!(qmpo_aux, last(sites) => randn(D, d, d, 1))
            new_qmpo_aux = copy(qmpo_aux)
            push!(qmpo, i => new_qmpo_aux)
        else
            for j in sites_aux
                push!(qmpo_aux, j => randn(d_aux, d_aux))
            end
            push!(qmpo_aux, i => randn(D, d, d, D))
            new_qmpo_aux = copy(qmpo_aux)
            push!(qmpo, i => new_qmpo_aux)
        end
        empty!(qmpo_aux)
    end
    QMpo(qmpo)
end

"""
$(TYPEDSIGNATURES)
"""
function is_left_normalized(ψ::QMps)
    all(
       I(size(A, 3)) ≈ @tensor Id[x, y] := conj(A[α, σ, x]) * A[α, σ, y] order = (α, σ)
       for A ∈ values(ψ.tensors)
    )
end

"""
$(TYPEDSIGNATURES)
"""
function is_right_normalized(ϕ::QMps)
    all(
        I(size(B, 1)) ≈ @tensor Id[x, y] := B[x, σ, α] * conj(B[y, σ, α]) order = (α, σ)
        for B in values(ϕ.tensors)
    )
end
