export
    contract_left,
    contract_up

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
        for v ∈ reverse(M.bot) B = contract_up(B, v) end   # contract_up  =>  attach_central_... ({T, 3} * {T, 2})
        B = contract_up(B, M.ctr)
        for v ∈ reverse(M.top) B = contract_up(B, v) end   # contract_up  =>  attach_central_...

        mps_li = left_nbrs_site(i, ϕ.sites)
        mpo_li = left_nbrs_site(i, ψ.sites)
        while mpo_li > mps_li
            B = contract_left(B, ψ[mpo_li].ctr)   # replace by reshape + attach_central_.... + reshape
            mpo_li = left_nbrs_site(mpo_li, ψ.sites)
        end
        push!(D, i => B)
    end
    QMps(D)
end

contract_up(A::Array{T, 3}, M::CentralOrDiagonal{T}) where T <: Real = attach_central_right(A, M)   # TODO: remove


function contract_left(A::Array{T, 3}, B::Matrix{T}) where T <: Real      # TODO: remove
    @matmul C[(x, y), u, r] := sum(σ) B[y, σ] * A[(x, σ), u, r] (σ ∈ 1:size(B, 2))
end

contract_left(A::Array{T, 3}, M::CentralTensor{T, 2}) where T <: Real = contract_left(A, Array(M))  # TODO: remove

contract_up(A::Array{T, 3}, B::Nothing) where T <: Real = A

function contract_up(A::Array{T, 3}, B::Matrix{T}) where T <: Real   # TODO: remove
    @tensor C[l, u, r] := B[u, σ] * A[l, σ, r]
end

function contract_up(A::Array{T, 3}, B::Array{T, 4}) where T <: Real
    @matmul C[(x, y), z, (b, a)] := sum(σ) B[y, z, a, σ] * A[x, σ, b]
end

function contract_up(A::Array{T, 3}, B::SiteTensor{T}) where T <: Real   # move to site tensor  {T, 3} * {T, 4} -> {T, 3}
    sal, _, sar = size(A)
    sbl, sbt, sbr = maximum.(B.projs[1:3])
    C = zeros(T, sal, sbl, sbt, sar, sbr)
    for (σ, lexp) ∈ enumerate(B.loc_exp)
        AA = @inbounds @view A[:, B.projs[4][σ], :]
        @inbounds C[:, B.projs[1][σ], B.projs[2][σ], :, B.projs[3][σ]] += lexp .* AA
    end
    @cast CC[(x, y), z, (b, a)] := C[x, y, z, b, a]
end


function contract_up(A::Array{T, 3}, B::VirtualTensor{T}) where T <: Real
    h = B.con
    if typeof(h) <: CentralTensor h = Array(h) end

    sal, _, sar = size(A)
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = B.projs
    C = zeros(T, sal, length(p_l), maximum(p_lt), maximum(p_rt), sar, length(p_r))

    @cast A4[x, k, l, y] := A[x, (k, l), y] (k ∈ 1:maximum(p_lb))

    for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
        AA = @inbounds @view A4[:, p_lb[l], p_rb[r], :]
        @inbounds C[:, l, p_lt[l], p_rt[r], :, r] += h[p_l[l], p_r[r]] .* AA
    end
    @cast CC[(x, y), (t1, t2), (b, a)] := C[x, y, t1, t2, b, a]
end

