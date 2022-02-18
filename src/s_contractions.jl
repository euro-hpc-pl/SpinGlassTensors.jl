export contract_left, contract_down, contract_up, dot, overlap_density_matrix

"""
$(TYPEDSIGNATURES)

"""
LinearAlgebra.dot(ψ::QMPS, ϕ::QMPS) = dot(MPS(ψ), MPS(ϕ))

"""
$(TYPEDSIGNATURES)

"""
LinearAlgebra.norm(ψ::QMPS) = sqrt(abs(dot(ψ, ψ)))

"""
$(TYPEDSIGNATURES)

"""
function _dot(ψ::QMPO{S}, ϕ::QMPS{S}, contract_func) where {S <: Real}
    D = Dict{Site, Tensor{S}}()
    for i ∈ ϕ.sites
        T = collect(ψ[i])
        TT = ϕ[i]
        for (_, v) ∈ T
             TT = contract_func(TT, v)
        end

        mps_li = _left_nbrs_site(i, ϕ.sites)
        mpo_li = _left_nbrs_site(i, ψ.sites)
        while mpo_li > mps_li
            TT = contract_left(TT, ψ[mpo_li][0])
            mpo_li = _left_nbrs_site(mpo_li, ψ.sites)
        end
        push!(D, i => TT)
    end
    QMPS(D)
end

"""
$(TYPEDSIGNATURES)

"""
LinearAlgebra.dot(ψ::QMPO{S}, ϕ::QMPS{S}) where {S <: Real} = _dot(ψ, ϕ, contract_up)

"""
$(TYPEDSIGNATURES)

"""
LinearAlgebra.dot(ϕ::QMPS{S}, ψ::QMPO{S}) where {S <: Real} = _dot(ψ, ϕ, contract_down)

"""
$(TYPEDSIGNATURES)

"""
function LinearAlgebra.dot(W::MPO, ϕ::QMPS)
    QMPS(Dict(i => contract_up(ϕ[i], A) for (i, A) ∈ enumerate(W)))
end

"""
$(TYPEDSIGNATURES)

"""
function LinearAlgebra.dot(ϕ::QMPS, W::MPO)
    QMPS(Dict(i => contract_down(ϕ[i], A) for (i, A) ∈ enumerate(W)))
end

"""
$(TYPEDSIGNATURES)

"""
function contract_left(A::AbstractArray{T, 3}, B::AbstractMatrix{T}) where T
    @cast C[(x, y), u, r] := sum(σ) B[y, σ] * A[(x, σ), u, r] (σ ∈ 1:size(B, 2))
    C
end

"""
$(TYPEDSIGNATURES)

"""
function contract_up(A::AbstractArray{T, 3}, B::AbstractArray{T, 2}) where T
    @tensor C[l, u, r] := B[u, σ] * A[l, σ, r]
    C
end

"""
$(TYPEDSIGNATURES)

"""
contract_down(A::AbstractArray{T, 3}, B::AbstractArray{T, 2}) where T = contract_up(A, B')

"""
$(TYPEDSIGNATURES)

"""
function contract_up(A::AbstractArray{T, 3}, B::AbstractArray{T, 4}) where T
    @matmul C[(x, y), z, (b, a)] := sum(σ) B[y, z, a, σ] * A[x, σ, b]
    C
end

"""
$(TYPEDSIGNATURES)

"""
function contract_down(A::AbstractArray{T, 3}, B::AbstractArray{T, 4}) where T
    contract_up(A, PermutedDimsArray(B, (1, 4, 3, 2)))
end

# TODO: improve performance
"""
$(TYPEDSIGNATURES)

"""
function contract_up(A::AbstractArray{T, 3}, B::SparseSiteTensor) where T
    sal, sac, sar = size(A)
    sbl, sbt, sbr = maximum.(B.projs[1:3])
    C = zeros(sal, sbl, sbt, sar, sbr)

    for (σ, lexp) ∈ enumerate(B.loc_exp)
        AA = @view A[:, B.projs[4][σ], :]
        C[:, B.projs[1][σ], B.projs[2][σ], :, B.projs[3][σ]] += lexp .* AA
    end
    @cast CC[(x, y), z, (b, a)] := C[x, y, z, b, a]
    CC
end

# TODO: improve performance
"""
$(TYPEDSIGNATURES)

"""
function contract_up(A::AbstractArray{T, 3}, B::SparseVirtualTensor) where T
    h = B.con
    sal, sac, sar = size(A)

    p_lb, p_l, p_lt, p_rb, p_r, p_rt = B.projs
    @cast A4[x, k, l, y] := A[x, (k, l), y] (k ∈ 1:maximum(p_lb))

    C = zeros(sal, length(p_l), maximum(p_rt), maximum(p_lt), sar, length(p_r))
    for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
        AA = @view A4[:, p_lb[l], p_rb[r], :]
        C[:, l, p_rt[r], p_lt[l], :, r] += h[p_l[l], p_r[r]] .* AA
    end
    @cast CC[(x, y), (t1, t2), (b, a)] := C[x, y, t1, t2, b, a]
    CC
end

"""
$(TYPEDSIGNATURES)

"""
function overlap_density_matrix(ϕ::QMPS, ψ::QMPS, k::Site)
    @assert ψ.sites == ϕ.sites
    C, D = ones(1, 1), ones(1, 1)
    for i ∈ ψ.sites
        if i < k
            A, B = ψ[i], ϕ[i]
            @tensor C[x, y] := conj(B)[β, σ, x] * C[β, α] * A[α, σ, y] order = (α, β, σ)
        end
    end
    for i ∈ reverse(ψ.sites)
        if i > k
            A, B = ψ[i], ϕ[i]
            @tensor D[x, y] := conj(B)[x, σ, β] * D[β, α] * A[y, σ, α] order = (α, β, σ)
        end
    end
    A, B = ψ[k], ϕ[k]
    @tensor E[x, y] := C[b, a] * conj(B)[b, x, β] * A[a, y, α] * D[β, α]
    E
end
