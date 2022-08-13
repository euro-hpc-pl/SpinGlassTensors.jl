export
    QMps,
    QMpo,
    local_dims,
    Site,
    Sites,
    Tensor,
    SparseSiteTensor,
    SparseVirtualTensor,
    SparseDiagonalTensor,
    SparseCentralTensor,
    dense_central_tensor,
    cuda_dense_central_tensor,
    IdentityQMps

abstract type AbstractEnvironment end
abstract type AbstractSparseTensor end

const Site = Union{Int, Rational{Int}}
const Sites = NTuple{N, Site} where N

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
    e1::Matrix{<:Real}
    e2::Matrix{<:Real}
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
@inline Base.size(tens::AbstractSparseTensor) = maximum.(tens.projs)

"""
$(TYPEDSIGNATURES)
"""
@inline function Base.setindex!(
    ket::AbstractTensorNetwork, A::AbstractArray, i::Site
)
    ket.tensors[i] = A
end

"""
$(TYPEDSIGNATURES)
"""
QMps(ϕ::AbstractMPS) = QMps(Dict(i => A for (i, A) ∈ enumerate(ϕ)))

"""
$(TYPEDSIGNATURES)
"""
QMpo(ϕ::AbstractMPO) = QMpo(Dict(i => Dict(0 => A) for (i, A) ∈ enumerate(ϕ)))
