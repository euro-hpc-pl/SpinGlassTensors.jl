export
    local_dims,
    IdentityQMps

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
