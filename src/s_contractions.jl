export
    contract_left,
    contract_down,
    contract_up,
    dot,
    overlap_density_matrix

# TODO  remove all connenctions with old mps
#LinearAlgebra.dot(ψ::QMps, ϕ::QMps) = dot(MPS(ψ), MPS(ϕ))
LinearAlgebra.norm(ψ::QMps) = sqrt(abs(dot(ψ, ψ)))

function LinearAlgebra.dot(ψ::QMps, ϕ::QMps)
    TTT = ones(1, 1)
    @assert ψ.sites == ϕ.sites
    for i ∈ ϕ.sites
        T = ϕ[i]
        TT = ψ[i]
        @tensor TTT[x, y] := conj(TT)[β, σ, x] * TTT[β, α] * T[α, σ, y] order = (α, β, σ)
    end
    tr(TTT)
end


function LinearAlgebra.dot(ψ::QMpo, ϕ::QMps)
    D = Dict{Site, Tensor}()
    for i ∈ reverse(ϕ.sites)
        T = sort(collect(ψ[i]), by = x -> x[begin])
        TT = ϕ[i]
        for (t, v) ∈ reverse(T) TT = contract_up(TT, v) end

        mps_li = _left_nbrs_site(i, ϕ.sites)
        mpo_li = _left_nbrs_site(i, ψ.sites)
        while mpo_li > mps_li
            TT = contract_left(TT, ψ[mpo_li][0])
            mpo_li = _left_nbrs_site(mpo_li, ψ.sites)
        end
        push!(D, i => TT)
    end
    QMps(D)
end

function LinearAlgebra.dot(ϕ::QMps, ψ::QMpo)
    D = Dict{Site, Tensor}()
    for i ∈ reverse(ϕ.sites)
        T = sort(collect(ψ[i]), by = x -> x[begin])
        TT = ϕ[i]
        for (t, v) ∈ T TT = contract_down(v, TT) end

        mps_li = _left_nbrs_site(i, ϕ.sites)
        mpo_li = _left_nbrs_site(i, ψ.sites)
        while mpo_li > mps_li
            TT = contract_left(TT, ψ[mpo_li][0])
            mpo_li = _left_nbrs_site(mpo_li, ψ.sites)
        end
        push!(D, i => TT)
    end
    QMps(D)
end

function LinearAlgebra.dot(W::MPO, ϕ::QMps)
    QMps(Dict(i => contract_up(ϕ[i], A) for (i, A) ∈ enumerate(W)))
end

function LinearAlgebra.dot(ϕ::QMps, W::MPO)
    QMps(Dict(i => contract_down(A, ϕ[i]) for (i, A) ∈ enumerate(W)))
end
Base.:(*)(W::QMpo, ψ::QMps) = dot(W, ψ)
Base.:(*)(ψ::QMps, W::QMpo) = dot(ψ, W)

function contract_left(A::AbstractArray{T, 3}, B::AbstractMatrix{T}) where T
    @matmul C[(x, y), u, r] := sum(σ) B[y, σ] * A[(x, σ), u, r] (σ ∈ 1:size(B, 2))
    C
end

function contract_up(A::AbstractArray{T, 3}, B::AbstractArray{T, 2}) where T
    @tensor C[l, u, r] := B[u, σ] * A[l, σ, r]
    C
end

function contract_down(A::AbstractArray{T, 2}, B::AbstractArray{T, 3}) where T
    @tensor C[l, d, r] := A[σ, d] * B[l, σ, r]
    C
end

function contract_up(A::AbstractArray{T, 3}, B::AbstractArray{T, 4}) where T
    @matmul C[(x, y), z, (b, a)] := sum(σ) B[y, z, a, σ] * A[x, σ, b]
    C
end

function contract_down(A::AbstractArray{T, 4}, B::AbstractArray{T, 3}) where T
    @matmul C[(x, y), z, (b, a)] := sum(σ) A[y, σ, a, z] * B[x, σ, b]
    C
end

# TODO: improve performance
function contract_up(A::AbstractArray{T, 3}, B::SparseSiteTensor) where T
    sal, _, sar = size(A)
    sbl, sbt, sbr = maximum.(B.projs[1:3])
    C = zeros(sal, sbl, sbt, sar, sbr)

    for (σ, lexp) ∈ enumerate(B.loc_exp)
        AA = @inbounds @view A[:, B.projs[4][σ], :]
        @inbounds C[:, B.projs[1][σ], B.projs[2][σ], :, B.projs[3][σ]] += lexp .* AA
    end
    @cast CC[(x, y), z, (b, a)] := C[x, y, z, b, a]
    CC
end

# TODO: improve performance
function contract_down(A::SparseSiteTensor, B::AbstractArray{T, 3}) where T
    sal, _, sar = size(B)
    sbl, _, sbt, sbr = maximum.(A.projs[1:4])
    C = zeros(sal, sbl, sbr, sar, sbt)

    for (σ, lexp) ∈ enumerate(A.loc_exp)
        AA = @inbounds @view B[:, A.projs[2][σ], :]
        @inbounds C[:, A.projs[1][σ], A.projs[4][σ], :, A.projs[3][σ]] += lexp .* AA
    end
    @cast CC[(x, y), z, (b, a)] := C[x, y, z, b, a]
    CC
end

# TODO: improve performance
function contract_up(A::AbstractArray{T, 3}, B::SparseVirtualTensor) where T
    h = B.con
    sal, _, sar = size(A)

    p_lb, p_l, p_lt, p_rb, p_r, p_rt = B.projs
    @cast A4[x, k, l, y] := A[x, (k, l), y] (k ∈ 1:maximum(p_lb))

    C = zeros(sal, length(p_l), maximum(p_rt), maximum(p_lt), sar, length(p_r))
    for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
        AA = @inbounds @view A4[:, p_lb[l], p_rb[r], :]
        @inbounds C[:, l, p_rt[r], p_lt[l], :, r] += h[p_l[l], p_r[r]] .* AA
    end
    @cast CC[(x, y), (t1, t2), (b, a)] := C[x, y, t1, t2, b, a]
    CC
end

function contract_down(A::SparseVirtualTensor, B::AbstractArray{T, 3}) where T
    h = A.con
    sal, _, sar = size(B)

    p_lb, p_l, p_lt, p_rb, p_r, p_rt = A.projs
    @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lb))

    C = zeros(sal, length(p_l), maximum(p_rt), maximum(p_lt), sar, length(p_r))
    for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
        BB = @inbounds @view B4[:, p_lt[l], p_rt[r], :]
        @inbounds C[:, l, p_rb[r], p_lb[l], :, r] += h[p_l[l], p_r[r]] .* BB
    end
    @cast CC[(x, y), (t1, t2), (b, a)] := C[x, y, t1, t2, b, a]
    CC
end

function overlap_density_matrix(ϕ::QMps, ψ::QMps, k::Site)
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
