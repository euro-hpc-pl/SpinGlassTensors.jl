@inline r2_over_r1(A) = size(A, 2) / size(A, 1)
@inline r1_over_r2(A) = 1 / r2_over_r1(A)

"""
Select optimal order of attaching matrices to L
"""
function attach_3_matrices_left(
    L::S, B2::Q, h::C, A2::Q
) where {S <: Array{T, 3}, Q <: Array{T, 2}, C <: Tensor{T, 2}} where T <: Real
    if r2_over_r1(h) <= r2_over_r1(B2) <= r2_over_r1(A2)
        L = contract_tensor3_matrix(L, h)  # [..., rc, ...] = [..., lc, ...] * [lc, rc]
        @tensor L[rfb, x, y] := L[lfb, x, y] * B2[lfb, rfb]
        @tensor L[x, y, rft] := L[x, y, lft] * A2[lft, rft]
    elseif r2_over_r1(h) <= r2_over_r1(A2) <= r2_over_r1(B2)
        L = contract_tensor3_matrix(L, h)  # [..., rc, ...] = [..., lc, ...] * [lc, rc]
        @tensor L[x, y, rft] := L[x, y, lft] * A2[lft, rft]
        @tensor L[rfb, x, y] := L[lfb, x, y] * B2[lfb, rfb]
    elseif r2_over_r1(A2) <= r2_over_r1(h) <= r2_over_r1(B2)
        @tensor L[x, y, rft] := L[x, y, lft] * A2[lft, rft]
        L = contract_tensor3_matrix(L, h)  # [..., rc, ...] = [..., lc, ...] * [lc, rc]
        @tensor L[rfb, x, y] := L[lfb, x, y] * B2[lfb, rfb]
    elseif r2_over_r1(A2) <= r2_over_r1(B2) <= r2_over_r1(h)
        @tensor L[x, y, rft] := L[x, y, lft] * A2[lft, rft]
        @tensor L[rfb, x, y] := L[lfb, x, y] * B2[lfb, rfb]
        L = contract_tensor3_matrix(L, h)  # [..., rc, ...] = [..., lc, ...] * [lc, rc]
    elseif r2_over_r1(B2) <= r2_over_r1(h) <= r2_over_r1(A2)
        @tensor L[rfb, x, y] := L[lfb, x, y] * B2[lfb, rfb]
        L = contract_tensor3_matrix(L, h)  # [..., rc, ...] = [..., lc, ...] * [lc, rc]
        @tensor L[x, y, rft] := L[x, y, lft] * A2[lft, rft]
    else # r2_over_r1(B2) <= r2_over_r1(A2) <= r2_over_r1(h)
        @tensor L[rfb, x, y] := L[lfb, x, y] * B2[lfb, rfb]
        @tensor L[x, y, rft] := L[x, y, lft] * A2[lft, rft]
        L = contract_tensor3_matrix(L, h)  # [..., rc, ...] = [..., lc, ...] * [lc, rc]
    end
    L
end

"""
Select optimal order of attaching matrices to R
"""
function attach_3_matrices_right(
    R::S, B2::Q, h::C, A2::Q
) where {S <: Array{T, 3}, Q <: Array{T, 2}, C <: Tensor{T, 2}} where T <: Real
    if r1_over_r2(h) <= r1_over_r2(B2) <= r1_over_r2(A2)
        R = contract_matrix_tensor3(h, R)  # [..., lc, ...] = [..., rc, ...] * [lc, rc]
        @tensor R[lfb, x, y] := R[rfb, x, y] * B2[lfb, rfb]
        @tensor R[x, y, lft] := R[x, y, rft] * A2[lft, rft]
    elseif r1_over_r2(h) <= r1_over_r2(A2) <= r1_over_r2(B2)
        R = contract_matrix_tensor3(h, R)  # [..., lc, ...] = [..., rc, ...] * [lc, rc]
        @tensor R[x, y, lft] := R[x, y, rft] * A2[lft, rft]
        @tensor R[lfb, x, y] := R[rfb, x, y] * B2[lfb, rfb]
    elseif r1_over_r2(A2) <= r1_over_r2(h) <= r1_over_r2(B2)
        @tensor R[x, y, lft] := R[x, y, rft] * A2[lft, rft]
        R = contract_matrix_tensor3(h, R)  # [..., lc, ...] = [..., rc, ...] * [lc, rc]
        @tensor R[lfb, x, y] := R[rfb, x, y] * B2[lfb, rfb]
    elseif r1_over_r2(A2) <= r1_over_r2(B2) <= r1_over_r2(h)
        @tensor R[x, y, lft] := R[x, y, rft] * A2[lft, rft]
        @tensor R[lfb, x, y] := R[rfb, x, y] * B2[lfb, rfb]
        R = contract_matrix_tensor3(h, R)  # [..., lc, ...] = [..., rc, ...] * [lc, rc]
    elseif r1_over_r2(B2) <= r1_over_r2(h) <= r1_over_r2(A2)
        @tensor R[lfb, x, y] := R[rfb, x, y] * B2[lfb, rfb]
        R = contract_matrix_tensor3(h, R)  # [..., lc, ...] = [..., rc, ...] * [lc, rc]
        @tensor R[x, y, lft] := R[x, y, rft] * A2[lft, rft]
    else # r1_over_r2(B2) <= r1_over_r2(A2) <= r1_over_r2(h)
        @tensor R[lfb, x, y] := R[rfb, x, y] * B2[lfb, rfb]
        @tensor R[x, y, lft] := R[x, y, rft] * A2[lft, rft]
        R = contract_matrix_tensor3(h, R)  # [..., lc, ...] = [..., rc, ...] * [lc, rc]
    end
    R
end

function attach_2_matrices(
    L::S, B2::Q, h::C, R::S
    ) where {S <: Array{T, 3}, Q <: Array{T, 2}, C <: Tensor{T, 2}} where T <: Real
    h1, h2 = size(h, 1), size(h, 2)
    b1, b2 = size(B2, 1), size(B2, 2)
    leg_list = [h1, h2, b1, b2]
    _, max_index = findmax(leg_list)
    biggest_leg = leg_list[max_index]

    if biggest_leg ∈ [h1, h2]
        if h1 >= h2
            L = contract_tensor3_matrix(L, h)
        else
            R = contract_matrix_tensor3(h, R)
        end
        if b1 >= b2
            @tensor L[rfb, x, y] := L[lfb, x, y] * B2[lfb, rfb]
        else
            @tensor R[x, y, lfb] := R[x, y, rfb] * B2[lfb, rfb]
        end
    else
        if b1 >= b2
            @tensor L[rfb, x, y] := L[lfb, x, y] * B2[lfb, rfb]
        else
            @tensor R[x, y, lfb] := R[x, y, rfb] * B2[lfb, rfb]
        end
        if h1 >= h2
            L = contract_tensor3_matrix(L, h)
        else
            R = contract_matrix_tensor3(h, R)
        end
    end
    @tensor LR[lft, rft] := L[lfb, lfh, lft] * R[rft, lfh, lfb]
end

function update_env_left(LE::S, A::S, M::VirtualTensor{T}, B::S) where {S <: Array{T, 3}} where T <: Real
    A, B, L = CuArray.((A, B, LE))
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    slb, srb = size(B, 1), size(B, 3)
    slt, srt = size(A, 1), size(A, 3)
    srcp = length(p_r)

    slcb, slc, slct = maximum(p_lb), maximum(p_l), maximum(p_lt)
    srcb, srct = maximum(p_rb), maximum(p_rt)
    prs = CuSparseMatrixCSR(T, p_rb, p_r, p_rt)
    ps = CuSparseMatrixCSC(T, p_lb, p_l, p_lt)

    batch_size = 2
    Lout = CUDA.zeros(T, srcp, srb, srt)
    @cast A2[(lt, lct), (rct, rt)] := A[lt, (lct, rct), rt] (lct ∈ 1:slct)

    lb_from = 1
    while lb_from <= slb
        lb_to = min(lb_from + batch_size - 1, slb)
        Lslc = L[lb_from:lb_to, :, :]
        Lslc = permutedims(Lslc, (2, 1, 3))  # [lcp, lb, lt]
        @cast Lslc[lcp, (lb, lt)] := Lslc[lcp, lb, lt]
        Lslc = ps * Lslc  # [(lcb, lc, lct), (lb, lt)]
        @cast Lslc[lcb, lc, lct, lb, lt] := Lslc[(lcb, lc, lct), (lb, lt)] (lcb ∈ 1:slcb, lc ∈ 1:slc, lt ∈ 1:slt)
        Lslc = permutedims(Lslc, (4, 1, 2, 5, 3)) #[lb, lcb, lc, lt, lct]
        @cast Lslc[(lb, lcb), lc, (lt, lct)] := Lslc[lb, lcb, lc, lt, lct]

        rb_from = 1
        while rb_from <= srb
            rb_to = min(rb_from + batch_size - 1, srb)
            Btemp = B[lb_from : lb_to, :, rb_from : rb_to]
            @cast B2[(lb, lcb), (rcb, rb)] := Btemp[lb, (lcb, rcb), rb] (lcb ∈ 1:slcb)
            Ltemp = attach_3_matrices_left(Lslc, B2, h, A2)
            @cast Ltemp[rcb, rb, rc, rct, rt] := Ltemp[(rcb, rb), rc, (rct, rt)] (rcb ∈ 1:srcb, rct ∈ 1:srct)
            Ltemp = permutedims(Ltemp, (1, 3, 4, 2, 5))  # [rcb, rc, rct, rb, rt]
            @cast Ltemp[(rcb, rc, rct), (rb, rt)] := Ltemp[rcb, rc, rct, rb, rt]
            Ltemp = prs * Ltemp  # [rcp, (rb, rt)]
            @cast Ltemp[rcp, rb, rt] := Ltemp[rcp, (rb, rt)] (rt ∈ 1:srt)
            Lout[:, rb_from : rb_to, :] += Ltemp
            rb_from = rb_to + 1
        end
        lb_from = lb_to + 1
    end
    Array(permutedims(Lout, (2, 1, 3)) ./ maximum(abs.(Lout))) #[rb, rcp, rt]
end

function update_env_right(RE::S, A::S, M::VirtualTensor{T}, B::S) where {S <: Array{T, 3}} where T <: Real
    A, B, R = CuArray.((A, B, RE))
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    slb, srb = size(B, 1), size(B, 3)
    slt, srt = size(A, 1), size(A, 3)
    slcp = length(p_l)

    slcb, slc, slct = maximum(p_lb), maximum(p_l), maximum(p_lt)
    srcb, src, srct = maximum(p_rb), maximum(p_r), maximum(p_rt)
    prs = CuSparseMatrixCSR(eltype(RE), p_lb, p_l, p_lt)
    ps = CuSparseMatrixCSC(eltype(RE), p_rb, p_r, p_rt)

    batch_size = 2
    Rout = CUDA.zeros(T, slcp, slt, slb)
    @cast A2[(lt, lct), (rct, rt)] := A[lt, (lct, rct), rt] (rct ∈ 1:srct)

    rb_from = 1
    while rb_from <= srb
        rb_to = min(rb_from + batch_size - 1, srb)
        Rslc = R[:, :, rb_from : rb_to]
        Rslc = permutedims(Rslc, (2, 3, 1))  # [rcp, rb, rt]
        @cast Rslc[rcp, (rb, rt)] := Rslc[rcp, rb, rt]
        Rslc = ps * Rslc  # [(rcb, rc, rct), (rb, rt)]
        @cast Rslc[rcb, rc, rct, rb, rt] := Rslc[(rcb, rc, rct), (rb, rt)] (rcb ∈ 1:srcb, rc ∈ 1:src, rt ∈ 1:srt)
        Rslc = permutedims(Rslc, (1, 4, 2, 3, 5))  # [rcb, rb, rc, rct, rt]
        @cast Rslc[(rcb, rb), rc, (rct, rt)] := Rslc[rcb, rb, rc, rct, rt]

        lb_from = 1
        while lb_from <= slb
            lb_to = min(lb_from + batch_size - 1, slb)
            Btemp = B[lb_from:lb_to, :, rb_from:rb_to]
            @cast B2[(lb, lcb), (rcb, rb)] := Btemp[lb, (lcb, rcb), rb] (rcb ∈ 1:srcb)
            Rtemp = attach_3_matrices_right(Rslc, B2, h, A2)
            @cast Rtemp[lb, lcb, lc, lt, lct] := Rtemp[(lb, lcb), lc, (lt, lct)] (lcb ∈ 1:slcb, lt ∈ 1:slt)
            Rtemp = permutedims(Rtemp, (2, 3, 5, 4, 1)) #[lcb, lc, lct, lb, lt]
            @cast Rtemp[(lcb, lc, lct), (lt, lb)] := Rtemp[lcb, lc, lct, lt, lb]
            Rtemp = prs * Rtemp  # [lcp, (lb, lt)]
            @cast Rtemp[lcp, lt, lb] := Rtemp[lcp, (lt, lb)] (lt ∈ 1:slt)
            Rout[:, :, lb_from:lb_to] += Rtemp
            lb_from = lb_to + 1
        end
        rb_from = rb_to + 1
    end
    Array(permutedims(Rout, (2, 1, 3)) ./ maximum(abs.(Rout))) #[lb, lcp, lt]
end

function project_ket_on_bra(LE::S, B::S, M::VirtualTensor{T}, RE::S) where {S <: Array{T, 3}} where T <: Real
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    B, L, R = CuArray.((B, LE, RE))
    sl1, sl3 = size(L, 1), size(L, 3)
    sr1, sr3 = size(R, 1), size(R, 3)

    slcb, slc, slct = maximum(p_lb), maximum(p_l), maximum(p_lt)
    srcb, src, srct = maximum(p_rb), maximum(p_r), maximum(p_rt)
    ps = CuSparseMatrixCSC(T, p_lb, p_l, p_lt)
    prs = CuSparseMatrixCSC(T, p_rb, p_r, p_rt)

    batch_size = 2
    @cast B2[(lb, lcb), (rcb, rb)] := B[lb, (lcb, rcb), rb] (rcb ∈ 1:srcb)
    LRout = CUDA.zeros(T, sl3, slct * srct, sr1)

    l_from = 1
    while l_from <= sl3
        l_to = min(l_from + batch_size - 1, sl3)
        Lslc = L[:, :, l_from : l_to]
        Lslc = permutedims(Lslc, (2, 1, 3))  # [lcp, lb, lt]
        @cast Lslc[lcp, (lb, lt)] := Lslc[lcp, lb, lt]
        Lslc = ps * Lslc #[(lcb, lc, lct), (lb, lt)]
        @cast Lslc[lcb, lc, lct, lb, lt] := Lslc[(lcb, lc, lct), (lb, lt)] (lcb ∈ 1:slcb, lc ∈ 1:slc, lb ∈ 1:sl1)
        Lslc = permutedims(Lslc, (4, 1, 2, 5, 3)) #[lb, lcb, lc, lt, lct]
        @cast Lslc[(lb, lcb), lc, (lt, lct)] := Lslc[lb, lcb, lc, lt, lct]
        r_from = 1

        while r_from <= sr1
            r_to = min(r_from + batch_size - 1, sr1)
            Rslc = R[r_from : r_to, :, :]
            Rslc = permutedims(Rslc, (2, 3, 1))  # [rcp, rb, rt]
            @cast Rslc[rcp, (rb, rt)] := Rslc[rcp, rb, rt]
            Rslc = prs * Rslc #[(rcb, rc, rct), (rb, rt)]
            @cast Rslc[rcb, rc, rct, rb, rt] := Rslc[(rcb, rc, rct), (rb, rt)] (rcb ∈ 1:srcb, rc ∈ 1:src, rb ∈ 1:sr3)
            Rslc = permutedims(Rslc, (3, 5, 2, 1, 4)) #[rct, rt, rc, rcb, rb]
            @cast Rslc[(rct, rt), rc, (rcb, rb)] := Rslc[rct, rt, rc, rcb, rb]
            LR = attach_2_matrices(Lslc, B2, h, Rslc)
            @cast LR[lt, (lct, rct), rt] := LR[(lt, lct), (rct, rt)] (lct ∈ 1:slct, rct ∈ 1:srct)
            LRout[l_from:l_to, :, r_from:r_to] += LR
            r_from = r_to + 1
        end
        l_from = l_to + 1
    end
    Array(LRout ./ maximum(abs.(LRout)))
end

function update_reduced_env_right(K::Array{T, 1}, RE::Array{T, 2}, M::VirtualTensor{T}, B::Array{T, 3}) where T <: Real
    K, B, RE = CuArray.((K, B, RE))
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs
    @cast K2[t1, t2] := K[(t1, t2)] (t1 ∈ 1:maximum(p_lt))
    @cast B2[l, lb, rb, r] := B[l, (lb, rb), r] (lb ∈ 1:maximum(p_lb))
    B2 = permutedims(B2, (2, 1, 3, 4))  # [lb, l, rb, r]
    @cast B2[(lb, l), (rb, r)] := B2[lb, l, rb, r]
    Rtemp = CuSparseMatrixCSC(T, p_rt, p_r, p_rb) * permutedims(RE, (2, 1))
    @cast Rtemp[rt, rc, (rb, b)] := Rtemp[(rt, rc, rb), b] (rb ∈ 1:maximum(p_rb), rc ∈ 1:maximum(p_r))
    Rtemp = attach_3_matrices_right(Rtemp, K2, h, B2)  # [lt, lc, (lb, l)]
    @cast Rtemp[(lt, lc, lb), l] := Rtemp[lt, lc, (lb, l)] (lb ∈ 1:maximum(p_lb))
    Array(permutedims(CuSparseMatrixCSR(T, p_lt, p_l, p_lb) * Rtemp, (2, 1)))
end

function contract_tensors43(B::VirtualTensor{T, 4}, A::Array{T, 3}) where T <: Real
    h = B.con
    if typeof(h) <: CentralTensor h = Array(h) end #TODO add better handling

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
