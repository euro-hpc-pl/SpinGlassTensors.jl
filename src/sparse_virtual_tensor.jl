r2over1(matrix) = size(matrix, 2) / size(matrix, 1)
r1over2(matrix) = size(matrix, 1) / size(matrix, 2)

function attach_3_matrices_left(L, B2, h, A2)
    # selecting optimal order of attaching matrices to L
    if r2over1(h) <= r2over1(B2) <= r2over1(A2)
        L = attach_central_left(L, h)  # [..., rc, ...] = [..., lc, ...] * [lc, rc]
        @tensor L[rfb, x, y] := L[lfb, x, y] * B2[lfb, rfb]
        @tensor L[x, y, rft] := L[x, y, lft] * A2[lft, rft]
    elseif r2over1(h) <= r2over1(A2) <= r2over1(B2)
        L = attach_central_left(L, h)  # [..., rc, ...] = [..., lc, ...] * [lc, rc]
        @tensor L[x, y, rft] := L[x, y, lft] * A2[lft, rft]
        @tensor L[rfb, x, y] := L[lfb, x, y] * B2[lfb, rfb]
    elseif r2over1(A2) <= r2over1(h) <= r2over1(B2)
        @tensor L[x, y, rft] := L[x, y, lft] * A2[lft, rft]
        L = attach_central_left(L, h)  # [..., rc, ...] = [..., lc, ...] * [lc, rc]
        @tensor L[rfb, x, y] := L[lfb, x, y] * B2[lfb, rfb]
    elseif r2over1(A2) <= r2over1(B2) <= r2over1(h)
        @tensor L[x, y, rft] := L[x, y, lft] * A2[lft, rft]
        @tensor L[rfb, x, y] := L[lfb, x, y] * B2[lfb, rfb]
        L = attach_central_left(L, h)  # [..., rc, ...] = [..., lc, ...] * [lc, rc]
    elseif r2over1(B2) <= r2over1(h) <= r2over1(A2)
        @tensor L[rfb, x, y] := L[lfb, x, y] * B2[lfb, rfb]
        L = attach_central_left(L, h)  # [..., rc, ...] = [..., lc, ...] * [lc, rc]
        @tensor L[x, y, rft] := L[x, y, lft] * A2[lft, rft]
    else # r2over1(B2) <= r2over1(A2) <= r2over1(h)
        @tensor L[rfb, x, y] := L[lfb, x, y] * B2[lfb, rfb]
        @tensor L[x, y, rft] := L[x, y, lft] * A2[lft, rft]
        L = attach_central_left(L, h)  # [..., rc, ...] = [..., lc, ...] * [lc, rc]
    end
    L 
end

function attach_3_matrices_right(R, B2, h, A2)
    # selecting optimal order of attaching matrices to R
    if r1over2(h) <= r1over2(B2) <= r1over2(A2)
        R = attach_central_right(R, h)  # [..., lc, ...] = [..., rc, ...] * [lc, rc]
        @tensor R[lfb, x, y] := R[rfb, x, y] * B2[lfb, rfb]
        @tensor R[x, y, lft] := R[x, y, rft] * A2[lft, rft]
    elseif r1over2(h) <= r1over2(A2) <= r1over2(B2)
        R = attach_central_right(R, h)  # [..., lc, ...] = [..., rc, ...] * [lc, rc]
        @tensor R[x, y, lft] := R[x, y, rft] * A2[lft, rft]
        @tensor R[lfb, x, y] := R[rfb, x, y] * B2[lfb, rfb]
    elseif r1over2(A2) <= r1over2(h) <= r1over2(B2)
        @tensor R[x, y, lft] := R[x, y, rft] * A2[lft, rft]
        R = attach_central_right(R, h)  # [..., lc, ...] = [..., rc, ...] * [lc, rc]
        @tensor R[lfb, x, y] := R[rfb, x, y] * B2[lfb, rfb]
    elseif r1over2(A2) <= r1over2(B2) <= r1over2(h)
        @tensor R[x, y, lft] := R[x, y, rft] * A2[lft, rft]
        @tensor R[lfb, x, y] := R[rfb, x, y] * B2[lfb, rfb]
        R = attach_central_right(R, h)  # [..., lc, ...] = [..., rc, ...] * [lc, rc]
    elseif r1over2(B2) <= r1over2(h) <= r1over2(A2)
        @tensor R[lfb, x, y] := R[rfb, x, y] * B2[lfb, rfb]
        R = attach_central_right(R, h)  # [..., lc, ...] = [..., rc, ...] * [lc, rc]
        @tensor R[x, y, lft] := R[x, y, rft] * A2[lft, rft]
    else # r1over2(B2) <= r1over2(A2) <= r1over2(h)
        @tensor R[lfb, x, y] := R[rfb, x, y] * B2[lfb, rfb]
        @tensor R[x, y, lft] := R[x, y, rft] * A2[lft, rft]
        R = attach_central_right(R, h)  # [..., lc, ...] = [..., rc, ...] * [lc, rc]
    end
    R 
end

function update_env_left(
    LE::S, A::S, M::T, B::S, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}

    A = CUDA.CuArray(A)
    B = CUDA.CuArray(B)
    L = CUDA.CuArray(LE)

    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    slb, srb = size(B, 1), size(B, 3)
    slcb, slc, slct = maximum(p_lb), maximum(p_l), maximum(p_lt)
    srcb, srct = maximum(p_rb), maximum(p_rt)

    @cast A2[(lt, lct), (rct, rt)] := A[lt, (lct, rct), rt] (lct ∈ 1:slct)
    @cast B2[(lb, lcb), (rcb, rb)] := B[lb, (lcb, rcb), rb] (lcb ∈ 1:slcb)

    L = permutedims(L, (2, 1, 3))  # [lcp, lb, lt]
    @cast L[lcp, (lb, lt)] := L[lcp, lb, lt]

    ps = projectors_to_sparse(p_lb, p_l, p_lt, typeof(L))
    L = ps * L  # [(lcb, lc, lct), (lb, lt)]

    @cast L[lcb, lc, lct, lb, lt] := L[(lcb, lc, lct), (lb, lt)] (lcb ∈ 1:slcb, lc ∈ 1:slc, lb ∈ 1:slb)
    L = permutedims(L, (4, 1, 2, 5, 3))  # [lb, lcb, lc, lt, lct]
    @cast L[(lb, lcb), lc, (lt, lct)] := L[lb, lcb, lc, lt, lct]

    L = attach_3_matrices_left(L, B2, h, A2)

    @cast L[rcb, rb, rc, rct, rt] := L[(rcb, rb), rc, (rct, rt)] (rcb ∈ 1:srcb, rct ∈ 1:srct)
    L = permutedims(L, (1, 3, 4, 2, 5)) #[rcb, rc, rct, rb, rt]
    @cast L[(rcb, rc, rct), (rb, rt)] := L[rcb, rc, rct, rb, rt]

    prs = projectors_to_sparse_transposed(p_rb, p_r, p_rt, typeof(L))
    L = prs * L  # [rcp, (rb, rt)]
    @cast L[rcp, rb, rt] := L[rcp, (rb, rt)] (rb ∈ 1:srb)
    Array(permutedims(L, (2, 1, 3)) ./ maximum(abs.(L)))  # [rb, rcp, rt]
end

function update_env_left(
    LE::S, A::S, M::T, B::S, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}

    A = CUDA.CuArray(A)
    B = CUDA.CuArray(B)
    L = CUDA.CuArray(LE)

    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    slb, srb = size(B, 1), size(B, 3)
    slcb, slc, slct = maximum(p_lb), maximum(p_l), maximum(p_lt)
    srcb, srct = maximum(p_rb), maximum(p_rt)

    @cast A2[(lt, lcb), (rcb, rt)] := A[lt, (lcb, rcb), rt] (lcb ∈ 1:slcb)
    @cast B2[(lb, lct), (rct, rb)] := B[lb, (lct, rct), rb] (lct ∈ 1:slct)

    L = permutedims(L, (2, 1, 3))  # [lcp, lb, lt]
    @cast L[lcp, (lb, lt)] := L[lcp, lb, lt]

    ps = projectors_to_sparse(p_lb, p_l, p_lt, typeof(L))
    L = ps * L  # [(lcb, lc, lct), (lb, lt)]

    @cast L[lcb, lc, lct, lb, lt] := L[(lcb, lc, lct), (lb, lt)] (lcb ∈ 1:slcb, lc ∈ 1:slc, lb ∈ 1:slb)
    L = permutedims(L, (4, 3, 2, 5, 1))  # [lb, lct, lc, lt, lcb]
    @cast L[(lb, lct), lc, (lt, lcb)] := L[lb, lct, lc, lt, lcb]

    L = attach_3_matrices_left(L, B2, h, A2)
    
    @cast L[rct, rb, rc, rcb, rt] := L[(rct, rb), rc, (rcb, rt)] (rct ∈ 1:srct, rcb ∈ 1:srcb)
    L = permutedims(L, (1, 3, 4, 2, 5)) #[rcb, rc, rct, rb, rt]
    @cast L[(rct, rc, rcb), (rb, rt)] := L[rct, rc, rcb, rb, rt]

    prs = projectors_to_sparse_transposed(p_rb, p_r, p_rt, typeof(L))
    L = prs * L  # [rcp, (rb, rt)]
    @cast L[rcp, rb, rt] := L[rcp, (rb, rt)] (rb ∈ 1:srb)
    Array(permutedims(L, (2, 1, 3)) ./ maximum(abs.(L)))  # [rb, rcp, rt]
end

function update_env_right(
    RE::S, A::S, M::T, B::S, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}

    A = CUDA.CuArray(A)
    B = CUDA.CuArray(B)
    R = CUDA.CuArray(RE)

    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    slt, srt = size(A, 1), size(A, 3)
    slb, srb = size(B, 1), size(B, 3)
    srcb, src, srct = maximum(p_rb), maximum(p_r), maximum(p_rt)
    slcb, slct = maximum(p_lb), maximum(p_lt)

    @cast A2[(lt, lct), (rct, rt)] := A[lt, (lct, rct), rt] (rct ∈ 1:srct)
    @cast B2[(lb, lcb), (rcb, rb)] := B[lb, (lcb, rcb), rb] (rcb ∈ 1:srcb)

    R = permutedims(R, (2, 3, 1))  # [rcp, rb, rt]
    @cast R[rcp, (rb, rt)] := R[rcp, rb, rt]

    ps = projectors_to_sparse(p_rb, p_r, p_rt, typeof(R))
    R = ps * R  # [(rcb, rc, rct), (rb, rt)]

    @cast R[rcb, rc, rct, rb, rt] := R[(rcb, rc, rct), (rb, rt)] (rcb ∈ 1:srcb, rc ∈ 1:src, rb ∈ 1:srb)
    R = permutedims(R, (1, 4, 2, 3, 5))  # [rcb, rb, rc, rct, rt]

    @cast R[(rcb, rb), rc, (rct, rt)] := R[rcb, rb, rc, rct, rt]

    R = attach_3_matrices_right(R, B2, h, A2)
    
    @cast R[lb, lcb, lc, lt, lct] := R[(lb, lcb), lc, (lt, lct)] (lb ∈ 1:slb, lt ∈ 1:slt)
    R = permutedims(R, (2, 3, 5, 4, 1)) #[lct, lc, lcb, lt, lb] #
    @cast R[(lcb, lc, lct), (lt, lb)] := R[lcb, lc, lct, lt, lb]
    
    prs = projectors_to_sparse_transposed(p_lb, p_l, p_lt, typeof(R)) #
    R = prs * R  # [rcp, (rt, rb)]
    @cast R[lcp, lt, lb] := R[lcp, (lt, lb)] (lb ∈ 1:slb)
    Array(permutedims(R, (2, 1, 3)) ./ maximum(abs.(R)))  # [rt, rcp, rb]
end

function update_env_right(
    RE::S, A::S, M::T, B::S, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}

    A = CUDA.CuArray(A)
    B = CUDA.CuArray(B)
    R = CUDA.CuArray(RE)

    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    slt, srt = size(A, 1), size(A, 3)
    slb, srb = size(B, 1), size(B, 3)
    srcb, src, srct = maximum(p_rb), maximum(p_r), maximum(p_rt)
    slcb, slct = maximum(p_lb), maximum(p_lt)

    @cast A2[(lt, lcb), (rcb, rt)] := A[lt, (lcb, rcb), rt] (rcb ∈ 1:srcb)
    @cast B2[(lb, lct), (rct, rb)] := B[lb, (lct, rct), rb] (rct ∈ 1:srct)

    R = permutedims(R, (2, 3, 1))  # [rcp, rb, rt]
    @cast R[rcp, (rb, rt)] := R[rcp, rb, rt]

    ps = projectors_to_sparse(p_rb, p_r, p_rt, typeof(R))
    R = ps * R  # [(rcb, rc, rct), (rb, rt)]

    @cast R[rcb, rc, rct, rb, rt] := R[(rcb, rc, rct), (rb, rt)] (rcb ∈ 1:srcb, rc ∈ 1:src, rb ∈ 1:srb)
    R = permutedims(R, (3, 4, 2, 1, 5))  # [rct, rb, rc, rcb, rt]

    @cast R[(rct, rb), rc, (rcb, rt)] := R[rct, rb, rc, rcb, rt]

    R = attach_3_matrices_right(R, B2, h, A2)
    
    @cast R[lb, lct, lc, lt, lcb] := R[(lb, lct), lc, (lt, lcb)] (lb ∈ 1:slb, lt ∈ 1:slt)
    R = permutedims(R, (5, 3, 2, 4, 1)) #[lct, lc, lcb, lt, lb] #
    @cast R[(lcb, lc, lct), (lt, lb)] := R[lcb, lc, lct, lt, lb]
    
    prs = projectors_to_sparse_transposed(p_lb, p_l, p_lt, typeof(R)) #
    R = prs * R  # [rcp, (rt, rb)]
    @cast R[lcp, lt, lb] := R[lcp, (lt, lb)] (lb ∈ 1:slb)
    Array(permutedims(R, (2, 1, 3)) ./ maximum(abs.(R)))  # [rt, rcp, rb]
end

"""
$(TYPEDSIGNATURES)
"""
function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    B = CUDA.CuArray(B)
    LE = CUDA.CuArray(LE)
    RE = CUDA.CuArray(RE)

    @cast B4[bl, bt1, bt2, br] := B[bl, (bt1, bt2), br] (bt1 ∈ 1:maximum(p_lb))

    slb = size(LE, 1)
    srt = size(RE, 1)

    LEn = permutedims(LE, (2, 1, 3))
    @cast LEn[lc, (lb, lt)] := LEn[lc, lb, lt]
    tLE = typeof(LE)
    ps = projectors_to_sparse(p_lb, p_l, p_lt, tLE)
    LL = ps * LEn

    @cast LL[lc1, lc, lc2, lb, lt] := LL[(lc1, lc, lc2), (lb, lt)] (lc1 ∈ 1:maximum(p_lb), lc ∈ 1:maximum(p_l), lb ∈ 1:slb)
    LL = permutedims(LL, (4, 1, 2, 5, 3))

    @cast LL[(lc, lc2), lt, (lc1, lb)] := LL[lc, lc2, lt, lc1, lb]
    LL = attach_central_left(LL, h)
    @cast LL[lc, lc2, lt, lc1, lb] := LL[(lc, lc2), lt, (lc1, lb)] (lc2 ∈ 1:maximum(p_lb), lb ∈ 1:maximum(p_lt))

    REn = permutedims(RE, (2, 3, 1))
    @cast REn[rc, (rb, rt)] := REn[rc, rb, rt]
    ps = projectors_to_sparse(p_rb, p_r, p_rt, tLE)
    RR = ps * REn

    @cast RR[rc1, rc, rc2, rb, rt] := RR[(rc1, rc, rc2), (rb, rt)] (rc1 ∈ 1:maximum(p_rb), rc ∈ 1:maximum(p_r), rt ∈ 1:srt)
    RR = permutedims(RR, (5, 3, 2, 4, 1))
    @tensor LR[lc1, lc2, rc2, rt] := LL[bl, bt1, rc, lc1, lc2] * RR[rt, rc2, rc, br, bt2] * B4[bl, bt1, bt2, br] order = (bl, bt1, bt2, br, rc)

    @cast LR[l, (c1, c2), b] := LR[l, c1, c2, b]

    Array(LR ./ maximum(abs.(LR)))
end

"""
$(TYPEDSIGNATURES)
"""
function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    B = CUDA.CuArray(B)
    LE = CUDA.CuArray(LE)
    RE = CUDA.CuArray(RE)

    @cast B4[bl, bt1, bt2, br] := B[bl, (bt1, bt2), br] (bt1 ∈ 1:maximum(p_lt))

    slb = size(LE, 1)
    srt = size(RE, 1)

    LEn = permutedims(LE, (2, 1, 3))
    @cast LEn[lc, (lb, lt)] := LEn[lc, lb, lt]
    tLE = typeof(LE)
    ps = projectors_to_sparse(p_lt, p_l, p_lb, tLE)
    LL = ps * LEn

    @cast LL[lc1, lc, lc2, lb, lt] := LL[(lc1, lc, lc2), (lb, lt)] (lc1 ∈ 1:maximum(p_lt), lc ∈ 1:maximum(p_l), lb ∈ 1:slb)
    LL = permutedims(LL, (4, 1, 2, 5, 3))

    @cast LL[(lc, lc2), lt, (lc1, lb)] := LL[lc, lc2, lt, lc1, lb]
    LL = attach_central_left(LL, h)
    @cast LL[lc, lc2, lt, lc1, lb] := LL[(lc, lc2), lt, (lc1, lb)] (lc2 ∈ 1:maximum(p_lt), lb ∈ 1:maximum(p_lb))

    REn = permutedims(RE, (2, 3, 1))
    @cast REn[rc, (rb, rt)] := REn[rc, rb, rt]
    ps = projectors_to_sparse(p_rb, p_r, p_rt, tLE)
    RR = ps * REn

    @cast RR[rc1, rc, rc2, rb, rt] := RR[(rc1, rc, rc2), (rb, rt)] (rc1 ∈ 1:maximum(p_rt), rc ∈ 1:maximum(p_r), rt ∈ 1:srt)
    RR = permutedims(RR, (5, 3, 2, 4, 1))
    @tensor LR[lc1, lb, rc2, rt] := LL[bl, bt1, rc, lc1, lb] * RR[rt, rc2, rc, br, bt2] * B4[bl, bt1, bt2, br] order = (bl, bt1, bt2, br, rc)

    @cast LR[l, (c1, c2), r] := LR[l, c1, c2, r]

    Array(LR ./ maximum(abs.(LR)))
end
