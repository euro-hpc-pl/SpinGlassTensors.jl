
# ./mps/rand.jl: This file provides methods to generate random MPS / MPO

function Base.rand(::Type{QMps{T}}, loc_dims::Dict, Dmax::Int=1) where T <: Real
    id = TensorMap{T}(keys(loc_dims) .=> rand.(T, Dmax, Dmax, values(loc_dims)))
    site_min, ld_min = minimum(loc_dims)
    site_max, ld_max = maximum(loc_dims)
    if site_min == site_max
        id[site_min] = rand(T, 1, 1, ld_min)
    else
        id[site_min] = rand(T, 1, Dmax, ld_min)
        id[site_max] = rand(T, Dmax, 1, ld_max)
    end
    QMps(id)
end

function Base.rand(::Type{QMpo{T}}, sites::Vector, D::Int, d::Int, sites_aux::Vector=[], d_aux::Int=0) where T <: Real
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
                    (j => rand(T, d_aux, d_aux) for j ∈ sites_aux)...) for i ∈ 2:length(sites)-1)...
        )
    )
end
