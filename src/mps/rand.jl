
# ./mps/rand.jl: This file provides methods to generate random MPS / MPO

function Base.rand(::Type{QMps{T}}, loc_dims::Dict, Dmax::Int=1; onGPU=true) where T <: Real
    _rand = (onGPU ? CUDA.rand : rand)
    id = TensorMap{T}(keys(loc_dims) .=> _rand.(T, Dmax, values(loc_dims), Dmax))
    site_min, ld_min = minimum(loc_dims)
    site_max, ld_max = maximum(loc_dims)
    id[site_min] = _rand.(T, 1, ld_min, Dmax)
    id[site_max] = _rand.(T, Dmax, ld_max, 1)
    QMps(id; onGPU=onGPU)
end

function Base.rand(
    ::Type{QMpo{T}}, sites::Vector, D::Int, d::Int, sites_aux::Vector=[], d_aux::Int=0; onGPU=false
) where T <:Real
    _rand = (onGPU ? CUDA.rand : rand)
    QMpo(
        MpoTensorMap{T}(
            1 => MpoTensor{T}(
                    1 => _rand(T, 1, d, d, D),
                    (j => _rand(T, d_aux, d_aux) for j ∈ sites_aux)...
            ),
            sites[end] => MpoTensor{T}(
                    sites[end] => _rand(T, D, d, d, 1),
                    (j => _rand(T, d_aux, d_aux) for j ∈ sites_aux)...,
            ),
            (i => MpoTensor{T}(
                    i => _rand(T, D, d, d, D),
                    (j => _rand(T, d_aux, d_aux) for j ∈ sites_aux)...) for i ∈ 2:length(sites)-1)...
        )
    )
end
