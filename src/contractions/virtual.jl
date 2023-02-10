@inline r2_over_r1(A) = size(A, 2) / size(A, 1)
@inline r1_over_r2(A) = 1 / r2_over_r1(A)


"""
Select optimal order of attaching matrices to RE
"""
function attach_3_matrices_right(
    RE::S, B2::Q, A2::Q, h::C
) where {S <: Tensor{R, 3}, Q <: Tensor{R, 2}, C <: Tensor{R, 2}} where R <: Real
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




function update_env_right(RE::S, A::S, M::VirtualTensor{R, 4}, B::S) where {S <: Tensor{R, 3}} where R <: Real
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    slb, srb = size(B, 1), size(B, 2)
    slt, srt = size(A, 1), size(A, 2)
    slcp = length(M.lp, p_l)

    slcb, srcb, src, srct = size(M.lp, p_lb), size(M.lp, p_rb), size(M.lp, p_r), size(M.lp, p_rt)

    device = Tuple(which_device(h))[1]
    prs = SparseCSC(R, M.lp, p_lb, p_lt, p_l, device)
    pls = SparseCSC(R, M.lp, p_rb, p_rt, p_r, device)

    batch_size = 2
    Rout = typeof(RE) <: CuArray ? CUDA.zeros(R, slb, slt, slcp) : zeros(R, slb, slt, slcp)
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


# function update_env_right(RE::S, A::S, M::VirtualTensor{R, 4}, B::S) where {S <: Tensor{R, 3}} where R <: Real
#     h = M.con
#     p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

#     slb, srb = size(B, 1), size(B, 2)
#     slt, srt = size(A, 1), size(A, 2)
#     slcp = length(M.lp, p_l)

#     slcb, srcb, src, srct = size(M.lp, p_lb), size(M.lp, p_rb), size(M.lp, p_r), size(M.lp, p_rt)

#     device = Tuple(which_device(h))[1]
#     prs = SparseCSC(R, M.lp, p_lb, p_lt, p_l, device)
#     pls = SparseCSC(R, M.lp, p_rb, p_rt, p_r, device)

#     batch_size = 2
#     Rout = typeof(RE) <: CuArray ? CUDA.zeros(R, slb, slt, slcp) : zeros(R, slb, slt, slcp)
#     @cast A2[lt, rt, lct, rct] := A[lt, rt, (lct, rct)] (rct ∈ 1:srct)
#     A2 = permutedims(A2, (1, 3, 2, 4))
#     @cast A2[(lt, lct), (rt, rct)] := A2[lt, lct, rt, rct]

#     rb_from = 1
#     while rb_from <= srb
#         rb_to = min(rb_from + batch_size - 1, srb)
#         @inbounds Rslc = RE[rb_from:rb_to, :, :]
#         @cast Rslc[(rb, rt), rcp] := Rslc[rb, rt, rcp]
#         Rslc = (pls * Rslc')'  # [(rcb, rc, rct), (rb, rt)]
#         @cast Rslc[rb, rt, rcb, rct, rc] := Rslc[(rb, rt), (rcb, rct, rc)] (rcb ∈ 1:srcb, rc ∈ 1:src, rt ∈ 1:srt)
#         Rslc = permutedims(Rslc, (1, 3, 2, 4, 5))  # [rb, rcb, rt, rct, rc]
#         @cast Rslc[(rb, rcb), (rt, rct), rc] := Rslc[rb, rcb, rt, rct, rc]

#         lb_from = 1
#         while lb_from <= slb
#             lb_to = min(lb_from + batch_size - 1, slb)
#             @inbounds Btemp = B[lb_from:lb_to, rb_from:rb_to, :]
#             @cast Btemp[lb, rb, lcb, rcb] := Btemp[lb, rb, (lcb, rcb)] (lcb ∈ 1:slcb)
#             Btemp = permutedims(Btemp, (1, 3, 2, 4))
#             @cast B2[(lb, lcb), (rb, rcb)] := Btemp[lb, lcb, rb, rcb]
#             Rtemp = attach_3_matrices_right(Rslc, B2, A2, h)
#             @cast Rtemp[lb, lcb, lt, lct, lc] := Rtemp[(lb, lcb), (lt, lct), lc] (lcb ∈ 1:slcb, lt ∈ 1:slt)
#             Rtemp = permutedims(Rtemp, (1, 3, 2, 4, 5))  # [lb, lt, lcb, lct, lc]
#             @cast Rtemp[(lb, lt), (lcb, lct, lc)] := Rtemp[lb, lt, lcb, lct, lc]
#             Rtemp = (prs' * Rtemp')'  # [lcp, (lb, lt)]
#             @cast Rtemp[lb, lt, lcp] := Rtemp[(lb, lt), lcp] (lt ∈ 1:slt)
#             @inbounds Rout[lb_from:lb_to, :, :] += Rtemp
#             lb_from = lb_to + 1
#         end
#         rb_from = rb_to + 1
#     end
#     Rout ./ maximum(abs.(Rout))  # [lb, lt, lcp]
# end



function update_env_left(LE::S, A::S, M::VirtualTensor{R, 4}, B::S) where {S <: Tensor{R, 3}} where R <: Real
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    slb, srb = size(B, 1), size(B, 2)
    slt, srt = size(A, 1), size(A, 2)
    slcb, slc, slct = size(M.lp, p_lb), size(M.lp, p_l), size(M.lp, p_lt)
    srcb, src, srct = size(M.lp, p_rb), size(M.lp, p_r), size(M.lp, p_rt)
    srcp = length(M.lp, p_r)

    device = typeof(LE) <: CuArray ? :GPU : :CPU
    Lout = typeof(LE) <: CuArray ? CUDA.zeros(R, srcp, srb, srt) : zeros(R, srcp, srb, srt)

    A = reshape(A, (slt, srt, slct, srct))
    B = reshape(B, (slb, srb, slcb, srcb))

    if slcb >= slct
        pls = SparseCSC(R, M.lp, p_lt, p_l, p_lb, device)
        prs = SparseCSC(R, M.lp, p_rt, p_r, p_rb, device)
        B2 = permutedims(B, (3, 1, 4, 2))  # [lcb, lb, rcb, rb]
        B2 = reshape(B2, (slcb * slb, srcb * srb))  # [(lcb, lb), (rcb, rb)]
        for ilt ∈ 1 : slt
            Lslc = LE[:, ilt, :]  # [lb, lcp]
            Lslc = pls * Lslc'  # [(lct, lc, lcb), lb]
            Lslc = reshape(Lslc, (slct * slc, slcb * slb))  # [(lct, lc), (lcb, lb)]
            Lslc = Lslc * B2  # [(lct, lc), (lcb, lb)] * [(lcb, lb), (rcb, rb)]
            Lslc = reshape(Lslc, (slct, slc, srcb * srb))  # [lct, lc, (rcb, rb)]
            Lslc = permutedims(Lslc, (1, 3, 2))  # [lct, (rcb, rb), lc]
            Lslc = contract_tensor3_matrix(Lslc, M.con)  # [lct, (rcb, rb), rc]
            Lslc = permutedims(Lslc, (1, 3, 2))  # [lct, rc, (rcb, rb)]
            Lslc = reshape(Lslc, (slct, src * srcb * srb))  # [lct, (rc, rcb, rb)]
            for irt ∈ 1 : srt
                A2 = A[ilt, irt, :, :]  # [lct, rct]
                Ltemp = A2' * Lslc  # [rct, (rc, rcb, rb)]
                Ltemp = reshape(Ltemp, (srct * src * srcb, srb))
                Ltemp = prs' * Ltemp  # [rcp, rb]
                Lout[:, :, irt] += Ltemp
            end
        end
    else
        pls = SparseCSC(R, M.lp, p_lb, p_l, p_lt, device)
        prs = SparseCSC(R, M.lp, p_rb, p_r, p_rt, device)
        A2 = permutedims(A, (3, 1, 4, 2))  # [lct, lt, rct, rt]
        A2 = reshape(A2, (slct * slt, srct * srt))  # [(lct, lt), (rct, rt)]
        for ilb ∈ 1 : slb
            Lslc = LE[ilb, :, :]  # [lt, lcp]
            Lslc = pls * Lslc'  # [(lcb, lc, lct), lt]
            Lslc = reshape(Lslc, (slcb * slc, slct * slt))  # [(lcb, lc), (lct, lt)]
            Lslc = Lslc * A2  # [(lcb, lc), (lct, lt)] * [(lct, lt), (rct, rt)]
            Lslc = reshape(Lslc, (slcb, slc, srct * srt))  # [lcb, lc, (rct, rt)]
            Lslc = permutedims(Lslc, (1, 3, 2))  # [lcb, (rct, rt), lc]
            Lslc = contract_tensor3_matrix(Lslc, M.con)  # [lcb, (rct, rt), rc]
            Lslc = permutedims(Lslc, (1, 3, 2))  # [lcb, rc, (rct, rt)]
            Lslc = reshape(Lslc, (slcb, src * srct * srt))  # [lcb, (rc, rct, rt)]
            for irb ∈ 1 : srb
                B2 = B[ilb, irb, :, :]  # [lcb, rcb]
                Ltemp = B2' * Lslc  # [rcb, (rc, rct, rt)]
                Ltemp = reshape(Ltemp, (srcb * src * srct, srt))
                Ltemp = prs' * Ltemp  # [rcp, rt]
                Lout[:, irb, :] += Ltemp
            end
        end
    end
    Lout = permutedims(Lout, (2, 3, 1))
    Lout ./ maximum(abs.(Lout))  # [rb, rt, rcp]
end


function project_ket_on_bra(LE::S, B::S, M::VirtualTensor{R, 4}, RE::S) where {S <: Tensor{R, 3}} where R <: Real
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    slb, slt = size(LE, 1), size(LE, 2)
    srb, srt = size(RE, 1), size(RE, 2)
    slcb, slc, slct = size(M.lp, p_lb), size(M.lp, p_l), size(M.lp, p_lt)
    srcb, src, srct = size(M.lp, p_rb), size(M.lp, p_r), size(M.lp, p_rt)

    device = typeof(LE) <: CuArray ? :GPU : :CPU
    LR = typeof(LE) <: CuArray ? CUDA.zeros(R, slt, srt, slct, srct) : zeros(R, slt, srt, slct, srct)

    pls = SparseCSC(R, M.lp, p_lt, p_l, p_lb, device)
    prs = SparseCSC(R, M.lp, p_rt, p_r, p_rb, device)

    B = reshape(B, (slb, srb, slcb, srcb))
    B2 = permutedims(B, (3, 1, 4, 2))  # [lcb, lb, rcb, rb]
    B2 = reshape(B2, (slcb * slb, srcb * srb))  # [(lcb, lb), (rcb, rb)]

    if slcb >= srcb
        for ilt ∈ 1 : slt
            Lslc = LE[:, ilt, :]  # [lb, lc]
            Lslc = pls * Lslc'  # [(lct, lc, lcb), lb]
            Lslc = reshape(Lslc, (slct * slc, slcb * slb))  # [(lct, lc), (lcb, lb)]
            Lslc = Lslc * B2  # [(lct, lc), (lcb, lb)] * [(lcb, lb), (rcb, rb)]
            Lslc = reshape(Lslc, (slct, slc, srcb * srb))  # [lct, lc, (rcb, rb)]
            Lslc = permutedims(Lslc, (1, 3, 2))  # [lct, (rcb, rb), lc]
            Lslc = contract_tensor3_matrix(Lslc, M.con)  # [lct, (rcb, rb), rc]
            Lslc = permutedims(Lslc, (1, 3, 2))  # [lct, rc, (rcb, rb)]
            Lslc = reshape(Lslc, (slct, src * srcb * srb))  # [lct, (rc, rcb, rb)]
            for irt ∈ 1 : srt
                Rslc = RE[:, irt, :]  # [rb, rc]
                Rslc = prs * Rslc'  # [(rct, rc, rcb), rb]
                Rslc = reshape(Rslc, (srct, src * srcb * srb))  # [rct, (rc, rcb, rb)]
                LR[ilt, irt, :, :] = Lslc * Rslc'  # [lct, rct]
            end
        end
    else
        for irt ∈ 1 : srt
            Rslc = RE[:, irt, :]  # [rb, rc]
            Rslc = prs * Rslc'  # [(rct, rc, rcb), rb]
            Rslc = reshape(Rslc, (srct * src, srcb * srb))  # [(rct, rc), (rcb, rb)]
            Rslc = Rslc * B2'  # [(rct, rc), (rcb, rb)] * [(lcb, lb), (rcb, rb)]'
            Rslc = reshape(Rslc, (srct, src, slcb * slb))  # [rct, rc, (lcb, lb)]
            Rslc = permutedims(Rslc, (1, 3, 2))  # [rct, (lcb, lb), rc]
            Rslc = contract_matrix_tensor3(M.con, Rslc)  # [rct, (lcb, lb), lc]
            Rslc = permutedims(Rslc, (1, 3, 2))  # [rct, lc, (lcb, lb)]
            Rslc = reshape(Rslc, (srct, slc * slcb * slb))  # [rct, (lc, lcb, lb)]
            for ilt ∈ 1 : slt
                Lslc = LE[:, ilt, :]  # [lb, lc]
                Lslc = pls * Lslc'  # [(lct, lc, lcb), lb]
                Lslc = reshape(Lslc, (slct, slc * slcb * slb))  # [lct, (lc, lcb, lb)]
                LR[ilt, irt, :, :] = Lslc * Rslc'  # [lct, rct]
            end
        end
    end
    reshape(LR, (slt, srt, slct * srct)) ./ maximum(abs.(LR))
end


function update_reduced_env_right(
    K::Tensor{R, 1}, RE::Tensor{R, 2}, M::VirtualTensor{R, 4}, B::Tensor{R, 3}
) where R <: Real
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    slcb, slc, slct = size(M.lp, p_lb), size(M.lp, p_l), size(M.lp, p_lt)
    srcb, src, srct = size(M.lp, p_rb), size(M.lp, p_r), size(M.lp, p_rt)

    @cast K2[t1, t2] := K[(t1, t2)] (t1 ∈ 1:slct)
    @cast B2[l, r, lb, rb] := B[l, r, (lb, rb)] (lb ∈ 1:slcb)
    B2 = permutedims(B2, (1, 3, 2, 4))  # [l, lb, r, rb]
    @cast B2[(l, lb), (r, rb)] := B2[l, lb, r, rb]
    
    device = Tuple(which_device(h))[1]
    pls = SparseCSC(R, M.lp, p_lb, p_lt, p_l, device)
    prs = SparseCSC(R, M.lp, p_rb, p_rt, p_r, device)
    Rtemp = permutedims((prs * RE'), (2, 1))
    @cast Rtemp[b, rb, rt, rc] := Rtemp[b, (rb, rt, rc)] (rb ∈ 1:srcb, rc ∈ 1:src)
    @cast Rtemp[(b, rb), rt, rc] := Rtemp[b, rb, rt, rc]
    Rtemp = attach_3_matrices_right(Rtemp, B2, K2, h)  # [(l, lb), lt, lc]
    @cast Rtemp[l, (lb, lt, lc)] := Rtemp[(l, lb), lt, lc] (lb ∈ 1:slcb)
    (pls' * Rtemp')'
end

# TODO rewrite this function, too many nasty patches now
function contract_tensors43(B::VirtualTensor{R, 4}, A::Tensor{R, 3}) where R <: Real
    h = B.con
    ongpu = typeof(A) <: CuArray

    h = Array(dense_central(h))
    A = Array(A)

    sal, sar, _  = size(A)
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = get_projector!.(Ref(B.lp), B.projs)

    C = zeros(R, sal, length(p_l), sar, length(p_r), maximum(p_lt), maximum(p_rt))

    @cast A4[x, y, k, l] := A[x, y, (k, l)] (k ∈ 1:maximum(p_lb))

    for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
        AA = @inbounds @view A4[:, :, p_lb[l], p_rb[r]]
        @inbounds C[:, l, :, r, p_lt[l], p_rt[r]] += h[p_l[l], p_r[r]] .* AA
    end
    @cast CC[(x, y), (b, a), (t1, t2)] := C[x, y, b, a, t1, t2]
    if ongpu
        CC = CuArray(CC)
    end
    CC
end
