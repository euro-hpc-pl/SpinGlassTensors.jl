export
    contract_left,
    contract_tensors43

LinearAlgebra.norm(ψ::QMps) = sqrt(abs(dot(ψ, ψ)))

Base.:(*)(ϕ::QMps, ψ::QMps) = dot(ϕ, ψ)
Base.:(*)(W::QMpo, ψ::QMps) = dot(W, ψ)

function LinearAlgebra.dot(ψ::QMps{T}, ϕ::QMps{T}) where T <: Real
    @assert ψ.sites == ϕ.sites
    C = ones(T, 1, 1)
    for i ∈ ϕ.sites
        A, B = ϕ[i], ψ[i]
        @tensor C[x, y] := conj(B)[β, σ, x] * C[β, α] * A[α, σ, y] order = (α, β, σ)
    end
    tr(C)
end

function LinearAlgebra.dot(ψ::QMpo{R}, ϕ::QMps{R}) where R <: Real
    D = TensorMap{R}()
    for i ∈ reverse(ϕ.sites)
        M, B = ψ[i], ϕ[i]
        for v ∈ reverse(M.bot) B = contract_matrix_tensor3(v, B) end
        B = contract_tensors43(M.ctr, B)
        for v ∈ reverse(M.top) B = contract_matrix_tensor3(v, B) end

        mps_li = left_nbrs_site(i, ϕ.sites)
        mpo_li = left_nbrs_site(i, ψ.sites)

        while mpo_li > mps_li
            st = size(B, 2)
            sl2 = size(ψ[mpo_li], 2)
            @cast B[l1, l2, (t, r)] := B[(l1, l2), t, r] (l2 ∈ 1:sl2)
            B = contract_matrix_tensor3(ψ[mpo_li], B)
            @cast B[(l1, l2), t, r] := B[l1, l2, (t, r)] (t ∈ 1:st)
            mpo_li = left_nbrs_site(mpo_li, ψ.sites)
        end
        push!(D, i => B)
    end
    QMps(D)
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
