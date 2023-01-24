"""
Select optimal order of attaching matrices to L
"""
function attach_3_matrices_left(
    L::S, B2::Q, A2::Q, h::C
) where {S <: CuArray{T, 3}, Q <: CuArray{T, 2}, C <: Tensor{T, 2}} where T <: Real
    #println("attach_3_matrices_left L = ", size(L), " h = " , size(h), " B2 = " , size(B2), " A2 = " , size(A2))
    #@time begin
    if r2_over_r1(h) <= r2_over_r1(B2) <= r2_over_r1(A2)
        L = contract_tensor3_matrix(L, h)  # [..., rc] = [..., lc] * [lc, rc]
        @tensor L[rfb, x, y] := B2[lfb, rfb] * L[lfb, x, y]
        @tensor L[x, rft, y] := A2[lft, rft] * L[x, lft, y]
    elseif r2_over_r1(h) <= r2_over_r1(A2) <= r2_over_r1(B2)
        L = contract_tensor3_matrix(L, h)  # [..., rc] = [..., lc] * [lc, rc]
        @tensor L[x, rft, y] := A2[lft, rft] * L[x, lft, y]
        @tensor L[rfb, x, y] := B2[lfb, rfb] * L[lfb, x, y]
    elseif r2_over_r1(A2) <= r2_over_r1(h) <= r2_over_r1(B2)
        @tensor L[x, rft, y] := A2[lft, rft] * L[x, lft, y]
        L = contract_tensor3_matrix(L, h)  # [..., rc] = [..., lc] * [lc, rc]
        @tensor L[rfb, x, y] := B2[lfb, rfb] * L[lfb, x, y]
    elseif r2_over_r1(A2) <= r2_over_r1(B2) <= r2_over_r1(h)
        @tensor L[x, rft, y] := A2[lft, rft] * L[x, lft, y]
        @tensor L[rfb, x, y] := B2[lfb, rfb] * L[lfb, x, y]
        L = contract_tensor3_matrix(L, h)  # [..., rc] = [..., lc] * [lc, rc]
    elseif r2_over_r1(B2) <= r2_over_r1(h) <= r2_over_r1(A2)
        @tensor L[rfb, x, y] := B2[lfb, rfb] * L[lfb, x, y]
        L = contract_tensor3_matrix(L, h)  # [..., rc] = [..., lc] * [lc, rc]
        @tensor L[x, rft, y] := A2[lft, rft] * L[x, lft, y]
    else # r2_over_r1(B2) <= r2_over_r1(A2) <= r2_over_r1(h)
        @tensor L[rfb, x, y] := B2[lfb, rfb] * L[lfb, x, y]
        @tensor L[x, rft, y] := A2[lft, rft] * L[x, lft, y]
        L = contract_tensor3_matrix(L, h)  # [..., rc] = [..., lc] * [lc, rc]
    #end
    end
    L
end

"""
Select optimal order of attaching matrices to R
"""
function attach_3_matrices_right(
    R::S, B2::Q, A2::Q, h::C
) where {S <: CuArray{T, 3}, Q <: CuArray{T, 2}, C <: Tensor{T, 2}} where T <: Real
    #println("attach_3_matrices_right R = ", size(R), " h = " , size(h), " B2 = " , size(B2), " A2 = " , size(A2))
    #@time begin
    if r1_over_r2(h) <= r1_over_r2(B2) <= r1_over_r2(A2)
        R = contract_matrix_tensor3(h, R)  # [..., lc] = [lc, rc] * [..., rc]
        @tensor R[lfb, x, y] := B2[lfb, rfb] * R[rfb, x, y]
        @tensor R[x, lft, y] := A2[lft, rft] * R[x, rft, y]
    elseif r1_over_r2(h) <= r1_over_r2(A2) <= r1_over_r2(B2)
        R = contract_matrix_tensor3(h, R)  # [..., lc] = [lc, rc] * [..., rc]
        @tensor R[x, lft, y] := A2[lft, rft] * R[x, rft, y]
        @tensor R[lfb, x, y] := B2[lfb, rfb] * R[rfb, x, y]
    elseif r1_over_r2(A2) <= r1_over_r2(h) <= r1_over_r2(B2)
        @tensor R[x, lft, y] := A2[lft, rft] * R[x, rft, y]
        R = contract_matrix_tensor3(h, R)  # [..., lc] = [lc, rc] * [..., rc]
        @tensor R[lfb, x, y] := B2[lfb, rfb] * R[rfb, x, y]
    elseif r1_over_r2(A2) <= r1_over_r2(B2) <= r1_over_r2(h)
        @tensor R[x, lft, y] := A2[lft, rft] * R[x, rft, y]
        @tensor R[lfb, x, y] := B2[lfb, rfb] * R[rfb, x, y]
        R = contract_matrix_tensor3(h, R)  # [..., lc] = [lc, rc] * [..., rc]
    elseif r1_over_r2(B2) <= r1_over_r2(h) <= r1_over_r2(A2)
        @tensor R[lfb, x, y] := B2[lfb, rfb] * R[rfb, x, y]
        R = contract_matrix_tensor3(h, R)  # [..., lc] = [lc, rc] * [..., rc]
        @tensor R[x, lft, y] := A2[lft, rft] * R[x, rft, y]
    else # r1_over_r2(B2) <= r1_over_r2(A2) <= r1_over_r2(h)
        @tensor R[lfb, x, y] := B2[lfb, rfb] * R[rfb, x, y]
        @tensor R[x, lft, y] := A2[lft, rft] * R[x, rft, y]
        R = contract_matrix_tensor3(h, R)  # [..., lc] = [lc, rc] * [..., rc]
    end
#end
    R
end

function attach_2_matrices(
    L::S, B2::Q, h::C, R::S
) where {S <: CuArray{T, 3}, Q <: CuArray{T, 2}, C <: Tensor{T, 2}} where T <: Real
    #println("attach_2_matrices = ", size(L), " h = " , size(h), " B2 = " , size(B2), " R = " , size(R))
    #@time begin
    if >=(size(B2)...)
        @tensor L[rfb, x, y] := B2[lfb, rfb] * L[lfb, x, y]
    else
        @tensor R[lfb, x, y] := B2[lfb, rfb] * R[rfb, x, y]
    end
    if >=(size(h)...)
        L = contract_tensor3_matrix(L, h)
    else
        R = contract_matrix_tensor3(h, R)
    end
#end
    @tensor LR[lft, rft] := L[fb, lft, fh] * R[fb, rft, fh]
end

function update_env_left(L::S, A::S, M::VirtualTensor{T}, B::S) where {S <: CuArray{T, 3}} where T <: Real
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    slb, srb = size(B, 1), size(B, 2)
    slt, srt = size(A, 1), size(A, 2)
    srcp = length(p_r)

    slcb, slc, slct = maximum(p_lb), maximum(p_l), maximum(p_lt)
    srcb, srct = maximum(p_rb), maximum(p_rt)
    prs = CuSparseMatrixCSC(T, p_rb, p_rt, p_r)
    pls = CuSparseMatrixCSC(T, p_lb, p_lt, p_l)

    batch_size = 2
    Lout = CUDA.zeros(T, srb, srt, srcp)
    @cast A2[lt, rt, lct, rct] := A[lt, rt, (lct, rct)] (lct ∈ 1:slct)
    A2 = permutedims(A2, (1, 3, 2, 4))
    @cast A2[(lt, lct), (rt, rct)] := A2[lt, lct, rt, rct]

    lb_from = 1
    while lb_from <= slb
        lb_to = min(lb_from + batch_size - 1, slb)
        Lslc = L[lb_from:lb_to, :, :]
        @cast Lslc[(lb, lt), lcp] := Lslc[lb, lt, lcp]
        Lslc = (pls * Lslc')'  # [(lcb, lct, lc), (lb, lt)]
        @cast Lslc[lb, lt, lcb, lct, lc] := Lslc[(lb, lt), (lcb, lct, lc)] (lcb ∈ 1:slcb, lc ∈ 1:slc, lt ∈ 1:slt)
        Lslc = permutedims(Lslc, (1, 3, 2, 4, 5))  # [lb, lcb, lt, lct, lc]
        @cast Lslc[(lb, lcb), (lt, lct), lc] := Lslc[lb, lcb, lt, lct, lc]

        rb_from = 1
        while rb_from <= srb
            rb_to = min(rb_from + batch_size - 1, srb)
            Btemp = B[lb_from:lb_to, rb_from:rb_to, :]
            @cast Btemp[lb, rb, lcb, rcb] := Btemp[lb, rb, (lcb, rcb)] (lcb ∈ 1:slcb)
            Btemp = permutedims(Btemp, (1, 3, 2, 4))
            @cast B2[(lb, lcb), (rb, rcb)] := Btemp[lb, lcb, rb, rcb]
            Ltemp = attach_3_matrices_left(Lslc, B2, A2, h)
            @cast Ltemp[rb, rcb, rt, rct, rc] := Ltemp[(rb, rcb), (rt, rct), rc] (rcb ∈ 1:srcb, rct ∈ 1:srct)
            Ltemp = permutedims(Ltemp, (2, 4, 5, 1, 3))  # [rcb, rct, rc, rb, rt]
            @cast Ltemp[(rcb, rct, rc), (rb, rt)] := Ltemp[rcb, rct, rc, rb, rt]
            Ltemp = (prs' * Ltemp)'  # [rcp, (rb, rt)]
            # Ltemp = permutedims(Ltemp, (2, 1))
            @cast Ltemp[rb, rt, rcp] := Ltemp[(rb, rt), rcp] (rt ∈ 1:srt)
            Lout[rb_from : rb_to, :, :] += Ltemp
            rb_from = rb_to + 1
        end
        lb_from = lb_to + 1
    end
    Lout ./ maximum(abs.(Lout))  # [rb, rt, rcp]
end

function update_env_right(R::S, A::S, M::VirtualTensor{T}, B::S) where {S <: CuArray{T, 3}} where T <: Real
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    slb, srb = size(B, 1), size(B, 2)
    slt, srt = size(A, 1), size(A, 2)
    slcp = length(p_l)

    slcb, srcb, src, srct = maximum(p_lb), maximum(p_rb), maximum(p_r), maximum(p_rt)
    prs = CuSparseMatrixCSC(T, p_lb, p_lt, p_l)
    pls = CuSparseMatrixCSC(T, p_rb, p_rt, p_r)

    batch_size = 2
    Rout = CUDA.zeros(T, slb, slt, slcp)
    @cast A2[lt, rt, lct, rct] := A[lt, rt, (lct, rct)] (rct ∈ 1:srct)
    A2 = permutedims(A2, (1, 3, 2, 4))
    @cast A2[(lt, lct), (rt, rct)] := A2[lt, lct, rt, rct]

    rb_from = 1
    while rb_from <= srb
        rb_to = min(rb_from + batch_size - 1, srb)
        Rslc = R[rb_from:rb_to, :, :]
        @cast Rslc[(rb, rt), rcp] := Rslc[rb, rt, rcp]
        Rslc = (pls * Rslc')'  # [(rcb, rc, rct), (rb, rt)]
        @cast Rslc[rb, rt, rcb, rct, rc] := Rslc[(rb, rt), (rcb, rct, rc)] (rcb ∈ 1:srcb, rc ∈ 1:src, rt ∈ 1:srt)
        Rslc = permutedims(Rslc, (1, 3, 2, 4, 5))  # [rb, rcb, rt, rct, rc]
        @cast Rslc[(rb, rcb), (rt, rct), rc] := Rslc[rb, rcb, rt, rct, rc]

        lb_from = 1
        while lb_from <= slb
            lb_to = min(lb_from + batch_size - 1, slb)
            Btemp = B[lb_from:lb_to, rb_from:rb_to, :]
            @cast Btemp[lb, rb, lcb, rcb] := Btemp[lb, rb, (lcb, rcb)] (lcb ∈ 1:slcb)
            Btemp = permutedims(Btemp, (1, 3, 2, 4))
            @cast B2[(lb, lcb), (rb, rcb)] := Btemp[lb, lcb, rb, rcb]
            Rtemp = attach_3_matrices_right(Rslc, B2, A2, h)
            @cast Rtemp[lb, lcb, lt, lct, lc] := Rtemp[(lb, lcb), (lt, lct), lc] (lcb ∈ 1:slcb, lt ∈ 1:slt)
            Rtemp = permutedims(Rtemp, (1, 3, 2, 4, 5))  # [lb, lt, lcb, lct, lc]
            @cast Rtemp[(lb, lt), (lcb, lct, lc)] := Rtemp[lb, lt, lcb, lct, lc]
            Rtemp = (prs' * Rtemp')'  # [lcp, (lb, lt)]
            # Rtemp = permutedims(Rtemp, (2, 1))
            @cast Rtemp[lb, lt, lcp] := Rtemp[(lb, lt), lcp] (lt ∈ 1:slt)
            Rout[lb_from:lb_to, :, :] += Rtemp
            lb_from = lb_to + 1
        end
        rb_from = rb_to + 1
    end
    Rout ./ maximum(abs.(Rout))  # [lb, lt, lcp]
end

function project_ket_on_bra(L::S, B::S, M::VirtualTensor{T}, R::S) where {S <: CuArray{T, 3}} where T <: Real
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    sl1, sl2 = size(L, 1), size(L, 2)
    sr1, sr2 = size(R, 1), size(R, 2)

    slcb, slc, slct = maximum(p_lb), maximum(p_l), maximum(p_lt)
    srcb, src, srct = maximum(p_rb), maximum(p_r), maximum(p_rt)
    pls = CuSparseMatrixCSC(T, p_lb, p_lt, p_l)
    prs = CuSparseMatrixCSC(T, p_rb, p_rt, p_r)

    batch_size = 1
    @cast Btemp[lb, rb, lcb, rcb] := B[lb, rb, (lcb, rcb)] (lcb ∈ 1:slcb)
    Btemp = permutedims(Btemp, (1, 3, 2, 4))
    @cast B2[(lb, lcb), (rb, rcb)] := Btemp[lb, lcb, rb, rcb]
    LRout = CUDA.zeros(T, sl2, slct, sr2, srct)

    l_from = 1
    while l_from <= sl2
        l_to = min(l_from + batch_size - 1, sl2)
        Lslc = L[:, l_from:l_to, :]
        @cast Lslc[(lb, lt), lcp] := Lslc[lb, lt, lcp]
        Lslc = (pls * Lslc')'  # [(lcb, lc, lct), (lb, lt)]
        @cast Lslc[lb, lt, lcb, lct, lc] := Lslc[(lb, lt), (lcb, lct, lc)] (lcb ∈ 1:slcb, lc ∈ 1:slc, lb ∈ 1:sl1)
        Lslc = permutedims(Lslc, (1, 3, 2, 4, 5))  # [lb, lcb, lt, lct, lc]
        @cast Lslc[(lb, lcb), (lt, lct), lc] := Lslc[lb, lcb, lt, lct, lc]

        r_from = 1
        while r_from <= sr2
            r_to = min(r_from + batch_size - 1, sr2)
            Rslc = R[:, r_from:r_to, :]
            @cast Rslc[(rb, rt), rcp] := Rslc[rb, rt, rcp]
            Rslc = (prs * Rslc')'  # [(rcb, rct, rc), (rb, rt)]
            @cast Rslc[rb, rt, rcb, rct, rc] := Rslc[(rb, rt), (rcb, rct, rc)] (rcb ∈ 1:srcb, rc ∈ 1:src, rb ∈ 1:sr1)
            Rslc = permutedims(Rslc, (1, 3, 2, 4, 5))  # [rb, rcb, rt, rct, rc]
            @cast Rslc[(rb, rcb), (rt, rct), rc] := Rslc[rb, rcb, rt, rct, rc]
            LR = attach_2_matrices(Lslc, B2, h, Rslc)
            @cast LR[lt, lct, rt, rct] := LR[(lt, lct), (rt, rct)] (lct ∈ 1:slct, rct ∈ 1:srct)
            LRout[l_from:l_to, :, r_from:r_to, :] += LR
            r_from = r_to + 1
        end
        l_from = l_to + 1
    end

    LRout = reshape(permutedims(LRout, (1, 3, 2, 4)), (sl2, sr2, slct * srct))
    LRout ./ maximum(abs.(LRout))
end

function update_reduced_env_right(K::CuArray{T, 1}, RE::CuArray{T, 2}, M::VirtualTensor{T}, B::CuArray{T, 3}) where T <: Real
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs
    @cast K2[t1, t2] := K[(t1, t2)] (t1 ∈ 1:maximum(p_lt))
    @cast B2[l, r, lb, rb] := B[l, r, (lb, rb)] (lb ∈ 1:maximum(p_lb))
    B2 = permutedims(B2, (1, 3, 2, 4))  # [l, lb, r, rb]
    @cast B2[(l, lb), (r, rb)] := B2[l, lb, r, rb]
    pls = CuSparseMatrixCSC(T, p_lb, p_lt, p_l)
    prs = CuSparseMatrixCSC(T, p_rb, p_rt, p_r)
    Rtemp = permutedims((prs * RE'), (2, 1))
    @cast Rtemp[b, rb, rt, rc] := Rtemp[b, (rb, rt, rc)] (rb ∈ 1:maximum(p_rb), rc ∈ 1:maximum(p_r))
    # Rtemp = permutedims(Rtemp, (4, 1, 2, 3))  # [b, rb, rt, rc]
    @cast Rtemp[(b, rb), rt, rc] := Rtemp[b, rb, rt, rc] 
    Rtemp = attach_3_matrices_right(Rtemp, B2, K2, h)  # [(l, lb), lt, lc]
    @cast Rtemp[l, (lb, lt, lc)] := Rtemp[(l, lb), lt, lc] (lb ∈ 1:maximum(p_lb))
    (pls' * Rtemp')'
end

# TODO rewrite this function, too many nasty patches now
function contract_tensors43(B::VirtualTensor{T, 4}, A::CuArray{T, 3}) where T <: Real
    h = B.con

    h = Array(dense_central(h))
    A = Array(A)

    sal, sar, _  = size(A)
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = Array.(B.projs)
    C = zeros(T, sal, length(p_l), sar, length(p_r), maximum(p_lt), maximum(p_rt))

    @cast A4[x, y, k, l] := A[x, y, (k, l)] (k ∈ 1:maximum(p_lb))

    for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
        AA = @inbounds @view A4[:, :, p_lb[l], p_rb[r]]
        @inbounds C[:, l, :, r, p_lt[l], p_rt[r]] += h[p_l[l], p_r[r]] .* AA
    end
    @cast CC[(x, y), (b, a), (t1, t2)] := C[x, y, b, a, t1, t2]
    CuArray(CC)
end
