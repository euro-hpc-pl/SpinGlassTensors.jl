

abstract type AbstractTensotNetwork{T <: Number} end

const Site = Union{Int, Rational{Int}}
const TensorMap = Dict{Site, DenseOrSparseTensor}
const NestedTensorMap = Dict{Site, TensorMap}

function Base.eltype(ten::Union{TensorMap, NestedTensorMap})
    promote_type(eltype.(values(ten))...)
end

for (N, T) ∈ ((:MPS, :TensorMap), (:MPO, :NestedTensorMap))
    @eval begin
        struct $N{T <: Number} <: AbstractTensorNetwork
            tensors::$T
            sites::Vector{Site}

            function $N(tensors)
                S = eltype(tensors)
                new{S}(tensors, sort(collect(keys(tensors))))
            end
        end
        export $N
    end
end

function local_dims(mpo::MPO, dir::Symbol)
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

function IdentityMPS(::Type{T}, loc_dims::Dict{Site, Int}, Dmax::Int=1)
    id = Dict{Site, Array{T, 3}}(
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
        for (site, ld) ∈ loc_dims id[site][1, :, 1] .= one(T) / sqrt(ld) end
    MPS(id)
end
IdentityMPS(::Type{T}, loc_dims::Dict, Dmax::Int=1) = IdentityMPS(Float64, loc_dims, Dmax)

Base.setindex!(ket::AbstractTensorNetwork, A::Array, i::Site) = ket.tensors[i] = A
