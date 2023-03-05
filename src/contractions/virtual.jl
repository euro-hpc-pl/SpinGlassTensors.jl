# virtual.jl: contractions with VirtualTensor on CPU and CUDA


# @memoize Dict 
alloc_undef(R, onGPU, shape) = onGPU ? CuArray{R}(undef, shape) : Array{R}(undef, shape)
alloc_zeros(R, onGPU, shape) = onGPU ? CUDA.zeros(R, shape) : zeros(R, shape)

function update_env_left(LE::S, A::S, M::VirtualTensor{R, 4}, B::S) where {S <: Tensor{R, 3}} where R <: Real
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    slb, srb = size(B, 1), size(B, 2)
    slt, srt = size(A, 1), size(A, 2)
    slcb, slc, slct = size(M.lp, p_lb), size(M.lp, p_l), size(M.lp, p_lt)
    srcb, src, srct = size(M.lp, p_rb), size(M.lp, p_r), size(M.lp, p_rt)
    srcp = length(M.lp, p_r)

    onGPU = typeof(LE) <: CuArray
    device = onGPU ? :GPU : :CPU
    Lout = alloc_zeros(R, onGPU, (srcp, srb, srt))

    A = reshape(A, (slt, srt, slct, srct))
    B = reshape(B, (slb, srb, slcb, srcb))

    if slcb * srct >= slct * srcb
        pls = SparseCSC(R, M.lp, p_lt, p_l, p_lb, device)
        prs = SparseCSC(R, M.lp, p_rt, p_r, p_rb, device)
        B2 = permutedims(B, (3, 1, 4, 2))  # [lcb, lb, rcb, rb]
        B2 = reshape(B2, (slcb * slb, srcb * srb))  # [(lcb, lb), (rcb, rb)]
        A2 = permutedims(A, (3, 4, 1, 2))  # [lt, rt, rct, lct]

        tmp1 = alloc_undef(R, onGPU, (slct * slc * slcb, slb)) 
        tmp2 = alloc_undef(R, onGPU, (slct * slc, srcb * srb)) 
        tmp3 = alloc_undef(R, onGPU, (slct, srcb * srb, slc)) 
        tmp5 = alloc_undef(R, onGPU, (slct, src, srcb * srb)) 
        tmp7 = alloc_undef(R, onGPU, (srct, src * srcb * srb))
        tmp8 = alloc_undef(R, onGPU, (srcp, srb))
        for ilt ∈ 1 : slt
            Lslc = LE[:, ilt, :]  # [lb, lcp]
            mul!(tmp1, pls, Lslc')  # [(lct, lc, lcb), lb]
            mul!(tmp2, reshape(tmp1, (slct * slc, slcb * slb)), B2)  # [(lct, lc), (lcb, lb)] * [(lcb, lb), (rcb, rb)]
            permutedims!(tmp3, reshape(tmp2, (slct, slc, srcb * srb)), (1, 3, 2))  # [lct, (rcb, rb), lc]
            tmp4 = contract_tensor3_matrix(tmp3, M.con)  # [lct, (rcb, rb), rc]
            permutedims!(tmp5, tmp4, (1, 3, 2))  # [lct, rc, (rcb, rb)]
            tmp6 = reshape(tmp5, (slct, src * srcb * srb))  # [lct, (rc, rcb, rb)]
            for irt ∈ 1 : srt
                mul!(tmp7, (@view A2[:, :, ilt, irt])', tmp6)
                mul!(tmp8, prs', reshape(tmp7, (srct * src * srcb, srb)))
                Lout[:, :, irt] += tmp8  # [rcp, rb]
            end
        end
    else
        pls = SparseCSC(R, M.lp, p_lb, p_l, p_lt, device)
        prs = SparseCSC(R, M.lp, p_rb, p_r, p_rt, device)
        A2 = permutedims(A, (3, 1, 4, 2))  # [lct, lt, rct, rt]
        A2 = reshape(A2, (slct * slt, srct * srt))  # [(lct, lt), (rct, rt)]
        B2 = permutedims(B, (3, 4, 1, 2))  # [lcb, rcb, lb, rb]

        tmp1 = alloc_undef(R, onGPU, (slcb * slc * slct, slt))
        tmp2 = alloc_undef(R, onGPU, (slcb * slc, srct * srt))
        tmp3 = alloc_undef(R, onGPU, (slcb, srct * srt, slc))
        tmp5 = alloc_undef(R, onGPU, (slcb, src, srct * srt))
        tmp7 = alloc_undef(R, onGPU, (srcb, src * srct * srt))
        tmp8 = alloc_undef(R, onGPU, (srcp, srt))
        for ilb ∈ 1 : slb
            Lslc = LE[ilb, :, :]  # [lt, lcp]
            mul!(tmp1, pls, Lslc')  # [(lcb, lc, lct), lt]
            mul!(tmp2, reshape(tmp1, (slcb * slc, slct * slt)), A2)  # [(lcb, lc), (lct, lt)] * [(lct, lt), (rct, rt)]
            permutedims!(tmp3, reshape(tmp2, (slcb, slc, srct * srt)), (1, 3, 2))  # [lcb, (rct, rt), lc]
            tmp4 = contract_tensor3_matrix(tmp3, M.con)  # [lcb, (rct, rt), rc]
            permutedims!(tmp5, tmp4, (1, 3, 2))  # [lcb, rc, (rct, rt)]
            tmp6 = reshape(tmp5, (slcb, src * srct * srt))  # [lcb, (rc, rct, rt)]
            for irb ∈ 1 : srb
                mul!(tmp7, (@view B2[:, :, ilb, irb])', tmp6)
                mul!(tmp8, prs', reshape(tmp7, (srcb * src * srct, srt)))
                Lout[:, irb, :] += tmp8  # [rcp, rb]
            end
        end
    end
    Lout = permutedims(Lout, (2, 3, 1))
    Lout ./ maximum(abs.(Lout))  # [rb, rt, rcp]
end


function proj_out(lp, k1, k2, k3, device)
    p1 = get_projector!(lp, k1, device)
    p2 = get_projector!(lp, k2, device)
    p3 = get_projector!(lp, k3, device)
    @assert length(p1) == length(p2) == length(p3)
    s1, s2 = size(lp, k1), size(lp, k2)
    p1 .+ s1 * (p2 .- 1) .+ s1 * s2 * (p3 .- 1)
end


function project_ket_on_bra(LE::S, B::S, M::VirtualTensor{R, 4}, RE::S) where {S <: Tensor{R, 3}} where R <: Real
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    slb, slt = size(LE, 1), size(LE, 2)
    srb, srt = size(RE, 1), size(RE, 2)
    slcb, slc, slct = size(M.lp, p_lb), size(M.lp, p_l), size(M.lp, p_lt)
    srcb, src, srct = size(M.lp, p_rb), size(M.lp, p_r), size(M.lp, p_rt)

    onGPU = typeof(LE) <: CuArray
    device = onGPU ? :GPU : :CPU
    LR = onGPU ? CUDA.zeros(R, slt, srt, slct, srct) : zeros(R, slt, srt, slct, srct)

    B = reshape(B, (slb, srb, slcb, srcb))
    B2 = permutedims(B, (1, 3, 2, 4))  # [lb, lcb, rb, rcb]
    B2 = reshape(B2, (slb * slcb, srb * srcb))  # [(lb, lcb), (rb, rcb)]

    if slcb >= srcb
        pls = proj_out(M.lp, p_lb, p_lt, p_l, device)
        prs = proj_out(M.lp, p_rb, p_r, p_rt, device)
        tmp1 = alloc_zeros(R, onGPU, (slb, slcb * slct * slc))
        tmp2 = alloc_undef(R, onGPU, (srb * srcb, slct * slc))
        tmp5 = alloc_undef(R, onGPU, (srcb * srb, src, slct))
        tmp7 = alloc_zeros(R, onGPU, (srb, srcb * src * srct))
        tmp8 = alloc_undef(R, onGPU, (slct, srct))
        for ilt ∈ 1 : slt
            tmp1[:, pls] = (@view LE[:, ilt, :])  # [lb, (lcb, lct, lc)]
            mul!(tmp2, B2', reshape(tmp1, (slb * slcb, slct * slc)))  # [(lb, lcb), (rb, rcb)]' * [(lb, lcb), (lct, lc)]
            tmp3 = reshape(tmp2, (srb * srcb, slct, slc))  # [(rb, rcb), lct, lc]
            tmp4 = contract_tensor3_matrix(tmp3, M.con)  # [(rb, rcb), lct, rc]
            permutedims!(tmp5, tmp4, (1, 3, 2))  # [(rb, rcb), rc, lct]
            tmp6 = reshape(tmp5, (srb * srcb * src, slct))  # [(rb, rcb, rc), lct]
            for irt ∈ 1 : srt
                tmp7[:, prs] = (@view RE[:, irt, :])  # [rb, (rcb, rc, rct)]
                mul!(tmp8, tmp6', reshape(tmp7, (srb * srcb * src, srct)))
                @inbounds LR[ilt, irt, :, :] = tmp8  # [lct, rct]
            end
        end
    else
        pls = proj_out(M.lp, p_lb, p_l, p_lt, device)
        prs = proj_out(M.lp, p_rb, p_rt, p_r, device)
        tmp1 = alloc_zeros(R, onGPU, (srb, srcb * srct * src))
        tmp2 = alloc_undef(R, onGPU, (slb * slcb, srct * src))
        tmp5 = alloc_undef(R, onGPU, (slb * slcb, slc, srct))
        tmp7 = alloc_zeros(R, onGPU, (slb, slcb * slc * slct))
        tmp8 = alloc_undef(R, onGPU, (slct, srct))
        for irt ∈ 1 : srt
            tmp1[:, prs] = (@view RE[:, irt, :])  # [rb, (rcb, rct, rc)]
            mul!(tmp2, B2, reshape(tmp1, (srb * srcb, srct * src)))  # [(lb, lcb), (rb, rcb)] * [(rb, rcb), (rct, rc)]
            tmp3 = reshape(tmp2, (slb * slcb, srct, src))  # [(lb, lcb), rct, rc]
            tmp4 = contract_matrix_tensor3(M.con, tmp3)  # [(lb, lcb), rct, lc]
            permutedims!(tmp5, tmp4, (1, 3, 2))  # [(lb, lcb), lc, rct]
            tmp6 = reshape(tmp5, (slb * slcb * slc, srct))  # [(lb, lcb, lc), rct]
            for ilt ∈ 1 : slt
                tmp7[:, pls] = (@view LE[:, ilt, :])  # [lb, (lcb, lc, lct)]
                mul!(tmp8, reshape(tmp7, (slb * slcb * slc, slct))', tmp6)
                @inbounds LR[ilt, irt, :, :] = tmp8  # [lct, rct]
            end
        end
    end
    reshape(LR, (slt, srt, slct * srct)) ./ maximum(abs.(LR))
end

# function project_ket_on_bra(LE::S, B::S, M::VirtualTensor{R, 4}, RE::S) where {S <: Tensor{R, 3}} where R <: Real
#     p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

#     slb, slt = size(LE, 1), size(LE, 2)
#     srb, srt = size(RE, 1), size(RE, 2)
#     slcb, slc, slct = size(M.lp, p_lb), size(M.lp, p_l), size(M.lp, p_lt)
#     srcb, src, srct = size(M.lp, p_rb), size(M.lp, p_r), size(M.lp, p_rt)

#     onGPU = typeof(LE) <: CuArray
#     device = onGPU ? :GPU : :CPU
#     LR = onGPU ? CUDA.zeros(R, slt, srt, slct, srct) : zeros(R, slt, srt, slct, srct)

#     pls = SparseCSC(R, M.lp, p_lt, p_l, p_lb, device)
#     prs = SparseCSC(R, M.lp, p_rt, p_r, p_rb, device)

#     B = reshape(B, (slb, srb, slcb, srcb))
#     B2 = permutedims(B, (3, 1, 4, 2))  # [lcb, lb, rcb, rb]
#     B2 = reshape(B2, (slcb * slb, srcb * srb))  # [(lcb, lb), (rcb, rb)]

#     RE = permutedims(RE, (1, 3, 2))
#     LE = permutedims(LE, (1, 3, 2))
#     if slcb >= srcb

#         tmp1 = alloc_undef(R, onGPU, (slct * slc * slcb, slb))
#         tmp2 = alloc_undef(R, onGPU, (slct * slc, srcb * srb))
#         tmp3 = alloc_undef(R, onGPU, (slct, srcb * srb, slc))
#         tmp5 = alloc_undef(R, onGPU, (slct, src, srcb * srb))
#         tmp7 = alloc_undef(R, onGPU, (srct * src * srcb, srb))
#         tmp8 = alloc_undef(R, onGPU, (slct, srct))
#         for ilt ∈ 1 : slt
#             mul!(tmp1, pls, (@view LE[:, :, ilt])')  # [(lct, lc, lcb), lb]
#             mul!(tmp2, reshape(tmp1, (slct * slc, slcb * slb)),  B2)  # [(lct, lc), (lcb, lb)] * [(lcb, lb), (rcb, rb)]
#             permutedims!(tmp3, reshape(tmp2, (slct, slc, srcb * srb)), (1, 3, 2))  # [lct, (rcb, rb), lc]
#             tmp4 = contract_tensor3_matrix(tmp3, M.con)  # [lct, (rcb, rb), rc]
#             permutedims!(tmp5, tmp4, (1, 3, 2))  # [lct, rc, (rcb, rb)]
#             tmp6 = reshape(tmp5, (slct, src * srcb * srb))  # [lct, (rc, rcb, rb)]
#             for irt ∈ 1 : srt
#                 mul!(tmp7, prs, (@view RE[:, :, irt])')  # [(rct, rc, rcb), rb]
#                 mul!(tmp8, tmp6, reshape(tmp7, (srct, src * srcb * srb))')
#                 LR[ilt, irt, :, :] .= tmp8  # [lct, rct]
#             end
#         end
#     else
#         tmp1 = alloc_undef(R, onGPU, (srct * src * srcb, srb))
#         tmp2 = alloc_undef(R, onGPU, (srct * src, slcb * slb))
#         tmp3 = alloc_undef(R, onGPU, (srct, slcb * slb, src))
#         tmp5 = alloc_undef(R, onGPU, (srct, slc, slcb * slb))
#         tmp7 = alloc_undef(R, onGPU, (slct * slc * slcb, slb))
#         tmp8 = alloc_undef(R, onGPU, (slct, srct))
#         for irt ∈ 1 : srt
#             mul!(tmp1, prs, (@view RE[:, :, irt])')  # [(rct, rc, rcb), rb]
#             mul!(tmp2, reshape(tmp1, (srct * src, srcb * srb)), B2')  # [(rct, rc), (rcb, rb)] * [(lcb, lb), (rcb, rb)]'
#             permutedims!(tmp3, reshape(tmp2, (srct, src, slcb * slb)) , (1, 3, 2))  # [rct, (lcb, lb), rc]
#             tmp4 = contract_matrix_tensor3(M.con, tmp3)  # [rct, (lcb, lb), lc]
#             permutedims!(tmp5, tmp4, (1, 3, 2))  # [rct, lc, (lcb, lb)]
#             tmp6 = reshape(tmp5, (srct, slc * slcb * slb))'  # [(lc, lcb, lb), rct]
#             for ilt ∈ 1 : slt
#                 mul!(tmp7, pls, (@view LE[:, :, ilt])')  # [(lct, lc, lcb), lb]
#                 mul!(tmp8, reshape(tmp7, (slct, slc * slcb * slb)), tmp6)
#                 LR[ilt, irt, :, :] .= tmp8  # [lct, rct]
#             end
#         end
#     end
#     reshape(LR, (slt, srt, slct * srct)) ./ maximum(abs.(LR))
# end


function update_env_right(RE::S, A::S, M::VirtualTensor{R, 4}, B::S) where {S <: Tensor{R, 3}} where R <: Real
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    slb, srb = size(B, 1), size(B, 2)
    slt, srt = size(A, 1), size(A, 2)
    slcb, slc, slct = size(M.lp, p_lb), size(M.lp, p_l), size(M.lp, p_lt)
    srcb, src, srct = size(M.lp, p_rb), size(M.lp, p_r), size(M.lp, p_rt)
    slcp = length(M.lp, p_l)

    onGPU = typeof(RE) <: CuArray
    device = onGPU ? :GPU : :CPU
    Rout = onGPU ? CUDA.zeros(R, slcp, slb, slt) : zeros(R, slcp, slb, slt)

    A = reshape(A, (slt, srt, slct, srct))
    B = reshape(B, (slb, srb, slcb, srcb))

    if srcb * slct >= srct * slcb
        pls = SparseCSC(R, M.lp, p_lt, p_l, p_lb, device)
        prs = SparseCSC(R, M.lp, p_rt, p_r, p_rb, device)
        B2 = permutedims(B, (3, 1, 4, 2))  # [lcb, lb, rcb, rb]
        B2 = reshape(B2, (slcb * slb, srcb * srb))  # [(lcb, lb), (rcb, rb)]
        A2 = permutedims(A, (3, 4, 1, 2))

        tmp1 = alloc_undef(R, onGPU, (srct * src * srcb, srb))
        tmp2 = alloc_undef(R, onGPU, (srct * src, slcb * slb))
        tmp3 = alloc_undef(R, onGPU, (srct, slcb * slb, src))
        tmp5 = alloc_undef(R, onGPU, (srct, slc, slcb * slb))
        tmp7 = alloc_undef(R, onGPU, (slct, slc * slcb * slb))
        tmp8 = alloc_undef(R, onGPU, (slcp, slb))
        for irt ∈ 1 : srt
            mul!(tmp1, prs, RE[:, irt, :]')  # [(rct, rc, rcb), rb]
            mul!(tmp2, reshape(tmp1, (srct * src, srcb * srb)), B2')  # [(rct, rc), (rcb, rb)] * [(lcb, lb), (rcb, rb)]'
            permutedims!(tmp3, reshape(tmp2, (srct, src, slcb * slb)), (1, 3, 2))  # [rct, (lcb, lb), rc]
            tmp4 = contract_matrix_tensor3(M.con, tmp3)  # [rct, (lcb, lb), lc]
            permutedims!(tmp5, tmp4, (1, 3, 2))  # [rct, lc, (lcb, lb)]
            tmp6 = reshape(tmp5, (srct, slc * slcb * slb))  # [rct, (lc, lcb, lb)]
            for ilt ∈ 1 : slt
                mul!(tmp7, (@view A2[:, :, ilt, irt]), tmp6)
                mul!(tmp8, pls', reshape(tmp7, (slct * slc * slcb, slb)))
                Rout[:, :, ilt] += tmp8  # [rcp, rb]
            end
        end
    else
        pls = SparseCSC(R, M.lp, p_lb, p_l, p_lt, device)
        prs = SparseCSC(R, M.lp, p_rb, p_r, p_rt, device)
        A2 = permutedims(A, (3, 1, 4, 2))  # [lcb, lb, rcb, rb]
        A2 = reshape(A2, (slct * slt, srct * srt))  # [(lct, lt), (rct, rt)]
        B2 = permutedims(B, (3, 4, 1, 2))

        tmp1 = alloc_undef(R, onGPU, (srcb * src * srct, srt))
        tmp2 = alloc_undef(R, onGPU, (srcb * src, slct * slt))
        tmp3 = alloc_undef(R, onGPU, (srcb, slct * slt, src))
        tmp5 = alloc_undef(R, onGPU, (srcb, slc, slct * slt))
        tmp7 = alloc_undef(R, onGPU, (slcb, slc * slct * slt))
        tmp8 = alloc_undef(R, onGPU, (slcp, slt))
        for irb ∈ 1 : srb
            mul!(tmp1, prs, RE[irb, :, :]')  # [(rcb, rc, rct), rt]
            mul!(tmp2, reshape(tmp1, (srcb * src, srct * srt)), A2')  # [(rcb, rc), (rct, rt)] * [(lct, lt), (rct, rt)]'
            permutedims!(tmp3, reshape(tmp2, (srcb, src, slct * slt)), (1, 3, 2))  # [rcb, (lct, lt), rc]
            tmp4 = contract_matrix_tensor3(M.con, tmp3)  # [rcb, (lct, lt), lc]
            permutedims!(tmp5, tmp4, (1, 3, 2))  # [rcb, lc, (lct, lt)]
            tmp6 = reshape(tmp5, (srcb, slc * slct * slt))  # [rcb, (lc, lct, lt)]
            for ilb ∈ 1 : slb
                mul!(tmp7, (@view B2[:, :, ilb, irb]), tmp6)
                mul!(tmp8, pls', reshape(tmp7, (slcb * slc * slct, slt)))
                Rout[:, ilb, :] += tmp8  # [lcp, lt]
            end
        end
    end
    Rout = permutedims(Rout, (2, 3, 1))  # [lb, lt, lcp]
    Rout ./ maximum(abs.(Rout))
end


function update_reduced_env_right(
    K::Tensor{R, 1}, RE::Tensor{R, 2}, M::VirtualTensor{R, 4}, B::Tensor{R, 3}
) where R <: Real
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    slb, srb = size(B, 1), size(B, 2)
    slcb, slc, slct = size(M.lp, p_lb), size(M.lp, p_l), size(M.lp, p_lt)
    srcb, src, srct = size(M.lp, p_rb), size(M.lp, p_r), size(M.lp, p_rt)

    onGPU = typeof(RE) <: CuArray
    device = onGPU ? :GPU : :CPU

    K = reshape(K, (slct, srct))  # [lct, rct]
    B = reshape(B, (slb, srb, slcb, srcb))  # [lb, rb, lcb, rcb]
    B2 = permutedims(B, (3, 1, 4, 2))  # [lcb, lb, rcb, rb]
    B2 = reshape(B2, (slcb * slb, srcb * srb))  # [(lcb, lb), (rcb, rb)]

    pls = SparseCSC(R, M.lp, p_lt, p_l, p_lb, device)
    prs = SparseCSC(R, M.lp, p_rt, p_r, p_rb, device)
    Rtemp = prs * RE'  # [(rct, rc, rcb), rb]
    if srcb * slct >= srct * slcb
        Rtemp = reshape(Rtemp, (srct * src, srcb * srb))  # [(rct, rc), (rcb, rb)]
        Rtemp = Rtemp * B2'  # [(rct, rc), (rcb, rb)] * [(lcb, lb), (rcb, rb)]'
        Rtemp = reshape(Rtemp, (srct, src, slcb * slb))  # [rct, rc, (lcb, lb)]
        Rtemp = permutedims(Rtemp, (1, 3, 2))  # [rct, (lcb, lb), rc]
        Rtemp = contract_matrix_tensor3(M.con, Rtemp)  # [rct, (lcb, lb), lc]
        Rtemp = permutedims(Rtemp, (1, 3, 2))  # [rct, lc, (lcb, lb)]
        Rtemp = reshape(Rtemp, (srct, slc * slcb * slb))  # [rct, (lc, lcb, lb)]
        Rtemp = K * Rtemp  # [lct, (lc, lcb, lb)]
    else
        Rtemp = reshape(Rtemp, (srct, src * srcb * srb))  # [rct, (rc, rcb, rb)]
        Rtemp = K * Rtemp  # [lct, (rc, rcb, rb)]
        Rtemp = reshape(Rtemp, (slct, src, srcb * srb))  # [lct, rc, (rcb, rb)]
        Rtemp = permutedims(Rtemp, (1, 3, 2))  # [lct, (rcb, rb), rc]
        Rtemp = contract_matrix_tensor3(M.con, Rtemp)  # [lct, (rcb, rb), lc]
        Rtemp = permutedims(Rtemp, (1, 3, 2))  # [lct, lc, (rcb, rb)]
        Rtemp = reshape(Rtemp, (slct * slc, srcb * srb))  # [(lct, lc), (rcb, rb)]
        Rtemp = Rtemp * B2'  # [(lct, lc), (lcb, lb)]
    end
    Rtemp = reshape(Rtemp, (slct * slc * slcb, slb))
    Rtemp = pls' * Rtemp  # [lcp, lb]
    Rtemp = permutedims(Rtemp, (2, 1))  # [lb, lcp]
    Rtemp ./ maximum(abs.(Rtemp))
end


function contract_tensors43(M::VirtualTensor{R, 4}, B::Tensor{R, 3}) where R <: Real
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    slb, srb = size(B, 1), size(B, 2)
    slcb, slc, slct = size(M.lp, p_lb), size(M.lp, p_l), size(M.lp, p_lt)
    srcb, src, srct = size(M.lp, p_rb), size(M.lp, p_r), size(M.lp, p_rt)
    slcp, srcp = length(M.lp, p_l), length(M.lp, p_r)

    B = reshape(B, (slb, srb, slcb, srcb))

    pls = SparseCSC(R, M.lp, p_lb, p_l, p_lt, :CPU)
    pls = typeof(B) <: CuArray ? CuArray(pls) : Array(pls)
    pls = reshape(pls, (slcb, slc, slct * slcp))
    pls = permutedims(pls, (3, 1, 2))  # [(slct, slcp), lcb, lc]

    prs = SparseCSC(R, M.lp, p_rb, p_r, p_rt, :CPU)
    prs = typeof(B) <: CuArray ? CuArray(prs) : Array(prs)
    prs = reshape(prs, (srcb, src, srct * srcp))
    prs = permutedims(prs, (3, 1, 2))  # [(rct, rcp), rcb, rc]

    if size(M.con, 1) <= size(M.con, 2)
        prs = contract_matrix_tensor3(M.con, prs)
    else
        pls = contract_tensor3_matrix(pls, M.con)
    end
    @tensor MB[l, lt, r, rt] := pls[lt, lb, c] * prs[rt, rb, c] * B[l, r, lb, rb]  order=(lb, c, rb)
    MB = reshape(MB, slb, slct, slcp, srb, srct, srcp)
    MB = permutedims(MB, (1, 3, 4, 6, 2, 5))
    reshape(MB, (slb * slcp, srb * srcp, slct * srct))
end
