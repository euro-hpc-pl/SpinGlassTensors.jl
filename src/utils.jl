
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
