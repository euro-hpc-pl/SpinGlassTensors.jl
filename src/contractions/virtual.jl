@inline r2_over_r1(A) = size(A, 2) / size(A, 1)
@inline r1_over_r2(A) = 1 / r2_over_r1(A)

"""
Select optimal order of attaching matrices to LE
"""
function attach_3_matrices_left(
    LE::S, B2::Q, A2::Q, h::C
) where {S <: CuArray{R, 3}, Q <: CuArray{R, 2}, C <: Tensor{R, 2}} where R <: Real
    if r2_over_r1(h) <= r2_over_r1(B2) <= r2_over_r1(A2)
        LE = contract_tensor3_matrix(LE, h)  # [..., rc] = [..., lc] * [lc, rc]
        @tensor LE[rfb, x, y] := B2[lfb, rfb] * LE[lfb, x, y]
        @tensor LE[x, rft, y] := A2[lft, rft] * LE[x, lft, y]
    elseif r2_over_r1(h) <= r2_over_r1(A2) <= r2_over_r1(B2)
        LE = contract_tensor3_matrix(LE, h)  # [..., rc] = [..., lc] * [lc, rc]
        @tensor LE[x, rft, y] := A2[lft, rft] * LE[x, lft, y]
        @tensor LE[rfb, x, y] := B2[lfb, rfb] * LE[lfb, x, y]
    elseif r2_over_r1(A2) <= r2_over_r1(h) <= r2_over_r1(B2)
        @tensor LE[x, rft, y] := A2[lft, rft] * LE[x, lft, y]
        LE = contract_tensor3_matrix(LE, h)  # [..., rc] = [..., lc] * [lc, rc]
        @tensor LE[rfb, x, y] := B2[lfb, rfb] * LE[lfb, x, y]
    elseif r2_over_r1(A2) <= r2_over_r1(B2) <= r2_over_r1(h)
        @tensor LE[x, rft, y] := A2[lft, rft] * LE[x, lft, y]
        @tensor LE[rfb, x, y] := B2[lfb, rfb] * LE[lfb, x, y]
        LE = contract_tensor3_matrix(LE, h)  # [..., rc] = [..., lc] * [lc, rc]
    elseif r2_over_r1(B2) <= r2_over_r1(h) <= r2_over_r1(A2)
        @tensor LE[rfb, x, y] := B2[lfb, rfb] * LE[lfb, x, y]
        LE = contract_tensor3_matrix(LE, h)  # [..., rc] = [..., lc] * [lc, rc]
        @tensor LE[x, rft, y] := A2[lft, rft] * LE[x, lft, y]
    else # r2_over_r1(B2) <= r2_over_r1(A2) <= r2_over_r1(h)
        @tensor LE[rfb, x, y] := B2[lfb, rfb] * LE[lfb, x, y]
        @tensor LE[x, rft, y] := A2[lft, rft] * LE[x, lft, y]
        LE = contract_tensor3_matrix(LE, h)  # [..., rc] = [..., lc] * [lc, rc]
    end
    LE
end

"""
Select optimal order of attaching matrices to RE
"""
function attach_3_matrices_right(
    RE::S, B2::Q, A2::Q, h::C
) where {S <: CuArray{R, 3}, Q <: CuArray{R, 2}, C <: Tensor{R, 2}} where R <: Real
    if r1_over_r2(h) <= r1_over_r2(B2) <= r1_over_r2(A2)
        RE = contract_matrix_tensor3(h, RE)  # [..., lc] = [lc, rc] * [..., rc]
        @tensor RE[lfb, x, y] := B2[lfb, rfb] * RE[rfb, x, y]
        @tensor RE[x, lft, y] := A2[lft, rft] * RE[x, rft, y]
    elseif r1_over_r2(h) <= r1_over_r2(A2) <= r1_over_r2(B2)
        RE = contract_matrix_tensor3(h, RE)  # [..., lc] = [lc, rc] * [..., rc]
        @tensor RE[x, lft, y] := A2[lft, rft] * RE[x, rft, y]
        @tensor RE[lfb, x, y] := B2[lfb, rfb] * RE[rfb, x, y]
    elseif r1_over_r2(A2) <= r1_over_r2(h) <= r1_over_r2(B2)
        @tensor RE[x, lft, y] := A2[lft, rft] * RE[x, rft, y]
        RE = contract_matrix_tensor3(h, RE)  # [..., lc] = [lc, rc] * [..., rc]
        @tensor RE[lfb, x, y] := B2[lfb, rfb] * RE[rfb, x, y]
    elseif r1_over_r2(A2) <= r1_over_r2(B2) <= r1_over_r2(h)
        @tensor RE[x, lft, y] := A2[lft, rft] * RE[x, rft, y]
        @tensor RE[lfb, x, y] := B2[lfb, rfb] * RE[rfb, x, y]
        RE = contract_matrix_tensor3(h, RE)  # [..., lc] = [lc, rc] * [..., rc]
    elseif r1_over_r2(B2) <= r1_over_r2(h) <= r1_over_r2(A2)
        @tensor RE[lfb, x, y] := B2[lfb, rfb] * RE[rfb, x, y]
        RE = contract_matrix_tensor3(h, RE)  # [..., lc] = [lc, rc] * [..., rc]
        @tensor RE[x, lft, y] := A2[lft, rft] * RE[x, rft, y]
    else # r1_over_r2(B2) <= r1_over_r2(A2) <= r1_over_r2(h)
        @tensor RE[lfb, x, y] := B2[lfb, rfb] * RE[rfb, x, y]
        @tensor RE[x, lft, y] := A2[lft, rft] * RE[x, rft, y]
        RE = contract_matrix_tensor3(h, RE)  # [..., lc] = [lc, rc] * [..., rc]
    end
    RE
end

function attach_2_matrices(
    LE::S, B2::Q, h::C, RE::S
) where {S <: CuArray{R, 3}, Q <: CuArray{R, 2}, C <: Tensor{R, 2}} where R <: Real
    if >=(size(B2)...)
        @tensor LE[rfb, x, y] := B2[lfb, rfb] * LE[lfb, x, y]
    else
        @tensor RE[lfb, x, y] := B2[lfb, rfb] * RE[rfb, x, y]
    end
    if >=(size(h)...)
        LE = contract_tensor3_matrix(LE, h)
    else
        RE = contract_matrix_tensor3(h, RE)
    end
    @tensor LR[lft, rft] := LE[fb, lft, fh] * RE[fb, rft, fh]
end

function update_env_left(LE::S, A::S, M::VirtualTensor{R}, B::S) where {S <: CuArray{R, 3}} where R <: Real
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    slb, srb = size(B, 1), size(B, 2)
    slt, srt = size(A, 1), size(A, 2)
    srcp = length(p_r)

    slcb, slc, slct = maximum(p_lb), maximum(p_l), maximum(p_lt)
    srcb, srct = maximum(p_rb), maximum(p_rt)
    prs = CuSparseMatrixCSC(R, p_rb, p_rt, p_r)
    pls = CuSparseMatrixCSC(R, p_lb, p_lt, p_l)

    batch_size = 2
    Lout = CUDA.zeros(R, srb, srt, srcp)
    @cast A2[lt, rt, lct, rct] := A[lt, rt, (lct, rct)] (lct ∈ 1:slct)
    A2 = permutedims(A2, (1, 3, 2, 4))
    @cast A2[(lt, lct), (rt, rct)] := A2[lt, lct, rt, rct]

    lb_from = 1
    while lb_from <= slb
        @inbounds lb_to = min(lb_from + batch_size - 1, slb)
        Lslc = LE[lb_from:lb_to, :, :]
        @cast Lslc[(lb, lt), lcp] := Lslc[lb, lt, lcp]
        Lslc = (pls * Lslc')'  # [(lcb, lct, lc), (lb, lt)]
        @cast Lslc[lb, lt, lcb, lct, lc] := Lslc[(lb, lt), (lcb, lct, lc)] (lcb ∈ 1:slcb, lc ∈ 1:slc, lt ∈ 1:slt)
        Lslc = permutedims(Lslc, (1, 3, 2, 4, 5))  # [lb, lcb, lt, lct, lc]
        @cast Lslc[(lb, lcb), (lt, lct), lc] := Lslc[lb, lcb, lt, lct, lc]

        rb_from = 1
        while rb_from <= srb
            rb_to = min(rb_from + batch_size - 1, srb)
            @inbounds Btemp = B[lb_from:lb_to, rb_from:rb_to, :]
            @cast Btemp[lb, rb, lcb, rcb] := Btemp[lb, rb, (lcb, rcb)] (lcb ∈ 1:slcb)
            Btemp = permutedims(Btemp, (1, 3, 2, 4))
            @cast B2[(lb, lcb), (rb, rcb)] := Btemp[lb, lcb, rb, rcb]
            Ltemp = attach_3_matrices_left(Lslc, B2, A2, h)
            @cast Ltemp[rb, rcb, rt, rct, rc] := Ltemp[(rb, rcb), (rt, rct), rc] (rcb ∈ 1:srcb, rct ∈ 1:srct)
            Ltemp = permutedims(Ltemp, (2, 4, 5, 1, 3))  # [rcb, rct, rc, rb, rt]
            @cast Ltemp[(rcb, rct, rc), (rb, rt)] := Ltemp[rcb, rct, rc, rb, rt]
            Ltemp = (prs' * Ltemp)'  # [rcp, (rb, rt)]
            @cast Ltemp[rb, rt, rcp] := Ltemp[(rb, rt), rcp] (rt ∈ 1:srt)
            @inbounds Lout[rb_from : rb_to, :, :] += Ltemp
            rb_from = rb_to + 1
        end
        lb_from = lb_to + 1
    end
    Lout ./ maximum(abs.(Lout))  # [rb, rt, rcp]
end

function update_env_right(RE::S, A::S, M::VirtualTensor{R}, B::S) where {S <: CuArray{R, 3}} where R <: Real
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    slb, srb = size(B, 1), size(B, 2)
    slt, srt = size(A, 1), size(A, 2)
    slcp = length(p_l)

    slcb, srcb, src, srct = maximum(p_lb), maximum(p_rb), maximum(p_r), maximum(p_rt)
    prs = CuSparseMatrixCSC(R, p_lb, p_lt, p_l)
    pls = CuSparseMatrixCSC(R, p_rb, p_rt, p_r)

    batch_size = 2
    Rout = CUDA.zeros(R, slb, slt, slcp)
    @cast A2[lt, rt, lct, rct] := A[lt, rt, (lct, rct)] (rct ∈ 1:srct)
    A2 = permutedims(A2, (1, 3, 2, 4))
    @cast A2[(lt, lct), (rt, rct)] := A2[lt, lct, rt, rct]

    rb_from = 1
    while rb_from <= srb
        rb_to = min(rb_from + batch_size - 1, srb)
        @inbounds Rslc = RE[rb_from:rb_to, :, :]
        @cast Rslc[(rb, rt), rcp] := Rslc[rb, rt, rcp]
        Rslc = (pls * Rslc')'  # [(rcb, rc, rct), (rb, rt)]
        @cast Rslc[rb, rt, rcb, rct, rc] := Rslc[(rb, rt), (rcb, rct, rc)] (rcb ∈ 1:srcb, rc ∈ 1:src, rt ∈ 1:srt)
        Rslc = permutedims(Rslc, (1, 3, 2, 4, 5))  # [rb, rcb, rt, rct, rc]
        @cast Rslc[(rb, rcb), (rt, rct), rc] := Rslc[rb, rcb, rt, rct, rc]

        lb_from = 1
        while lb_from <= slb
            lb_to = min(lb_from + batch_size - 1, slb)
            @inbounds Btemp = B[lb_from:lb_to, rb_from:rb_to, :]
            @cast Btemp[lb, rb, lcb, rcb] := Btemp[lb, rb, (lcb, rcb)] (lcb ∈ 1:slcb)
            Btemp = permutedims(Btemp, (1, 3, 2, 4))
            @cast B2[(lb, lcb), (rb, rcb)] := Btemp[lb, lcb, rb, rcb]
            Rtemp = attach_3_matrices_right(Rslc, B2, A2, h)
            @cast Rtemp[lb, lcb, lt, lct, lc] := Rtemp[(lb, lcb), (lt, lct), lc] (lcb ∈ 1:slcb, lt ∈ 1:slt)
            Rtemp = permutedims(Rtemp, (1, 3, 2, 4, 5))  # [lb, lt, lcb, lct, lc]
            @cast Rtemp[(lb, lt), (lcb, lct, lc)] := Rtemp[lb, lt, lcb, lct, lc]
            Rtemp = (prs' * Rtemp')'  # [lcp, (lb, lt)]
            @cast Rtemp[lb, lt, lcp] := Rtemp[(lb, lt), lcp] (lt ∈ 1:slt)
            @inbounds Rout[lb_from:lb_to, :, :] += Rtemp
            lb_from = lb_to + 1
        end
        rb_from = rb_to + 1
    end
    Rout ./ maximum(abs.(Rout))  # [lb, lt, lcp]
end

function project_ket_on_bra(LE::S, B::S, M::VirtualTensor{R}, RE::S) where {S <: CuArray{R, 3}} where R <: Real
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    sl1, sl2 = size(LE, 1), size(LE, 2)
    sr1, sr2 = size(RE, 1), size(RE, 2)

    slcb, slc, slct = maximum(p_lb), maximum(p_l), maximum(p_lt)
    srcb, src, srct = maximum(p_rb), maximum(p_r), maximum(p_rt)
    pls = CuSparseMatrixCSC(R, p_lb, p_lt, p_l)
    prs = CuSparseMatrixCSC(R, p_rb, p_rt, p_r)

    batch_size = 1
    @cast Btemp[lb, rb, lcb, rcb] := B[lb, rb, (lcb, rcb)] (lcb ∈ 1:slcb)
    Btemp = permutedims(Btemp, (1, 3, 2, 4))
    @cast B2[(lb, lcb), (rb, rcb)] := Btemp[lb, lcb, rb, rcb]
    LRout = CUDA.zeros(R, sl2, slct, sr2, srct)

    l_from = 1
    while l_from <= sl2
        l_to = min(l_from + batch_size - 1, sl2)
        @inbounds Lslc = LE[:, l_from:l_to, :]
        @cast Lslc[(lb, lt), lcp] := Lslc[lb, lt, lcp]
        Lslc = (pls * Lslc')'  # [(lcb, lc, lct), (lb, lt)]
        @cast Lslc[lb, lt, lcb, lct, lc] := Lslc[(lb, lt), (lcb, lct, lc)] (lcb ∈ 1:slcb, lc ∈ 1:slc, lb ∈ 1:sl1)
        Lslc = permutedims(Lslc, (1, 3, 2, 4, 5))  # [lb, lcb, lt, lct, lc]
        @cast Lslc[(lb, lcb), (lt, lct), lc] := Lslc[lb, lcb, lt, lct, lc]

        r_from = 1
        while r_from <= sr2
            r_to = min(r_from + batch_size - 1, sr2)
            @inbounds Rslc = RE[:, r_from:r_to, :]
            @cast Rslc[(rb, rt), rcp] := Rslc[rb, rt, rcp]
            Rslc = (prs * Rslc')'  # [(rcb, rct, rc), (rb, rt)]
            @cast Rslc[rb, rt, rcb, rct, rc] := Rslc[(rb, rt), (rcb, rct, rc)] (rcb ∈ 1:srcb, rc ∈ 1:src, rb ∈ 1:sr1)
            Rslc = permutedims(Rslc, (1, 3, 2, 4, 5))  # [rb, rcb, rt, rct, rc]
            @cast Rslc[(rb, rcb), (rt, rct), rc] := Rslc[rb, rcb, rt, rct, rc]
            LR = attach_2_matrices(Lslc, B2, h, Rslc)
            @cast LR[lt, lct, rt, rct] := LR[(lt, lct), (rt, rct)] (lct ∈ 1:slct, rct ∈ 1:srct)
            @inbounds LRout[l_from:l_to, :, r_from:r_to, :] += LR
            r_from = r_to + 1
        end
        l_from = l_to + 1
    end

    LRout = reshape(permutedims(LRout, (1, 3, 2, 4)), sl2, sr2, slct * srct)
    LRout ./ maximum(abs.(LRout))
end

function update_reduced_env_right(K::CuArray{R, 1}, RE::CuArray{R, 2}, M::VirtualTensor{R}, B::CuArray{R, 3}) where R <: Real
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs
    @cast K2[t1, t2] := K[(t1, t2)] (t1 ∈ 1:maximum(p_lt))
    @cast B2[l, r, lb, rb] := B[l, r, (lb, rb)] (lb ∈ 1:maximum(p_lb))
    B2 = permutedims(B2, (1, 3, 2, 4))  # [l, lb, r, rb]
    @cast B2[(l, lb), (r, rb)] := B2[l, lb, r, rb]
    pls = CuSparseMatrixCSC(R, p_lb, p_lt, p_l)
    prs = CuSparseMatrixCSC(R, p_rb, p_rt, p_r)
    Rtemp = permutedims((prs * RE'), (2, 1))
    @cast Rtemp[b, rb, rt, rc] := Rtemp[b, (rb, rt, rc)] (rb ∈ 1:maximum(p_rb), rc ∈ 1:maximum(p_r))
    @cast Rtemp[(b, rb), rt, rc] := Rtemp[b, rb, rt, rc]
    Rtemp = attach_3_matrices_right(Rtemp, B2, K2, h)  # [(l, lb), lt, lc]
    @cast Rtemp[l, (lb, lt, lc)] := Rtemp[(l, lb), lt, lc] (lb ∈ 1:maximum(p_lb))
    (pls' * Rtemp')'
end

# TODO rewrite this function, too many nasty patches now
function contract_tensors43(B::VirtualTensor{R, 4}, A::CuArray{R, 3}) where R <: Real
    h = B.con

    h = Array(dense_central(h))
    A = Array(A)

    sal, sar, _  = size(A)
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = Array.(B.projs)
    C = zeros(R, sal, length(p_l), sar, length(p_r), maximum(p_lt), maximum(p_rt))

    @cast A4[x, y, k, l] := A[x, y, (k, l)] (k ∈ 1:maximum(p_lb))

    for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
        AA = @inbounds @view A4[:, :, p_lb[l], p_rb[r]]
        @inbounds C[:, l, :, r, p_lt[l], p_rt[r]] += h[p_l[l], p_r[r]] .* AA
    end
    @cast CC[(x, y), (b, a), (t1, t2)] := C[x, y, b, a, t1, t2]
    CuArray(CC)
end
