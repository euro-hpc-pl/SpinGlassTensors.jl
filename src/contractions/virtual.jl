# virtual.jl: contractions with VirtualTensor on CPU and CUDA

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

    if slcb * srct >= slct * srcb
        pls = SparseCSC(R, M.lp, p_lt, p_l, p_lb, device)
        prs = SparseCSC(R, M.lp, p_rt, p_r, p_rb, device)
        B2 = permutedims(B, (3, 1, 4, 2))  # [lcb, lb, rcb, rb]
        B2 = reshape(B2, (slcb * slb, srcb * srb))  # [(lcb, lb), (rcb, rb)]
        A2 = permutedims(A, (3, 4, 1, 2))  # [lt, rt, rct, lct]

        Ltemp1 = typeof(LE) <: CuArray ? CuArray{R}(undef, (srct, src * srcb * srb)) : Array{R}(undef, (srct, src * srcb * srb))
        Ltemp2 = typeof(LE) <: CuArray ? CuArray{R}(undef, (srcp, srb)) : Array{R}(undef, (srcp, srb))
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
                mul!(Ltemp1, (@view A2[:, :, ilt, irt])', Lslc)
                mul!(Ltemp2, prs', reshape(Ltemp1, (srct * src * srcb, srb)))
                # Ltemp1 = (@view A2[:, :, ilt, irt])' * Lslc  # [rct, (rc, rcb, rb)]  
                # Ltemp2 = prs' * reshape(Ltemp1, (srct * src * srcb, srb))
                Lout[:, :, irt] += Ltemp2 # [rcp, rb]
            end
        end
    else
        pls = SparseCSC(R, M.lp, p_lb, p_l, p_lt, device)
        prs = SparseCSC(R, M.lp, p_rb, p_r, p_rt, device)
        A2 = permutedims(A, (3, 1, 4, 2))  # [lct, lt, rct, rt]
        A2 = reshape(A2, (slct * slt, srct * srt))  # [(lct, lt), (rct, rt)]
        B2 = permutedims(B, (3, 4, 1, 2))  # [lcb, rcb, lb, rb]
        
        Ltemp1 = typeof(LE) <: CuArray ? CuArray{R}(undef, (srcb, src * srct * srt)) : Array{R}(undef, (srcb, src * srct * srt))
        Ltemp2 = typeof(LE) <: CuArray ? CuArray{R}(undef, (srcp, srt)) : Array{R}(undef, (srcp, srt))
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
                mul!(Ltemp1, (@view B2[:, :, ilb, irb])', Lslc)
                mul!(Ltemp2, prs', reshape(Ltemp1, (srcb * src * srct, srt)))
                Lout[:, irb, :] += Ltemp2 # [rcp, rb]
                # Ltemp = (@view B2[:, :, ilb, irb])' * Lslc  # [rcb, (rc, rct, rt)]
                # Ltemp = reshape(Ltemp, (srcb * src * srct, srt))
                # Lout[:, irb, :] += prs' * Ltemp  # [rcp, rt]
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

    RE = permutedims(RE, (1, 3, 2))
    LE = permutedims(LE, (1, 3, 2))
    if slcb >= srcb
        for ilt ∈ 1 : slt
            Lslc = pls * (@view LE[:, :, ilt])'  # [(lct, lc, lcb), lb]
            Lslc = reshape(Lslc, (slct * slc, slcb * slb))  # [(lct, lc), (lcb, lb)]
            Lslc = Lslc * B2  # [(lct, lc), (lcb, lb)] * [(lcb, lb), (rcb, rb)]
            Lslc = reshape(Lslc, (slct, slc, srcb * srb))  # [lct, lc, (rcb, rb)]
            Lslc = permutedims(Lslc, (1, 3, 2))  # [lct, (rcb, rb), lc]
            Lslc = contract_tensor3_matrix(Lslc, M.con)  # [lct, (rcb, rb), rc]
            Lslc = permutedims(Lslc, (1, 3, 2))  # [lct, rc, (rcb, rb)]
            Lslc = reshape(Lslc, (slct, src * srcb * srb))  # [lct, (rc, rcb, rb)]
            for irt ∈ 1 : srt
                Rslc = prs * (@view RE[:, :, irt])'  # [(rct, rc, rcb), rb]
                Rslc = reshape(Rslc, (srct, src * srcb * srb))  # [rct, (rc, rcb, rb)]
                LR[ilt, irt, :, :] = Lslc * Rslc'  # [lct, rct]
            end
        end
    else
        for irt ∈ 1 : srt
            Rslc = prs * (@view RE[:, :, irt])'  # [(rct, rc, rcb), rb]
            Rslc = reshape(Rslc, (srct * src, srcb * srb))  # [(rct, rc), (rcb, rb)]
            Rslc = Rslc * B2'  # [(rct, rc), (rcb, rb)] * [(lcb, lb), (rcb, rb)]'
            Rslc = reshape(Rslc, (srct, src, slcb * slb))  # [rct, rc, (lcb, lb)]
            Rslc = permutedims(Rslc, (1, 3, 2))  # [rct, (lcb, lb), rc]
            Rslc = contract_matrix_tensor3(M.con, Rslc)  # [rct, (lcb, lb), lc]
            Rslc = permutedims(Rslc, (1, 3, 2))  # [rct, lc, (lcb, lb)]
            Rslc = reshape(Rslc, (srct, slc * slcb * slb))  # [rct, (lc, lcb, lb)]
            for ilt ∈ 1 : slt
                Lslc = pls * (@view LE[:, :, ilt])'  # [(lct, lc, lcb), lb]
                Lslc = reshape(Lslc, (slct, slc * slcb * slb))  # [lct, (lc, lcb, lb)]
                LR[ilt, irt, :, :] = Lslc * Rslc'  # [lct, rct]
            end
        end
    end
    reshape(LR, (slt, srt, slct * srct)) ./ maximum(abs.(LR))
end


function update_env_right(RE::S, A::S, M::VirtualTensor{R, 4}, B::S) where {S <: Tensor{R, 3}} where R <: Real
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    slb, srb = size(B, 1), size(B, 2)
    slt, srt = size(A, 1), size(A, 2)
    slcb, slc, slct = size(M.lp, p_lb), size(M.lp, p_l), size(M.lp, p_lt)
    srcb, src, srct = size(M.lp, p_rb), size(M.lp, p_r), size(M.lp, p_rt)
    slcp = length(M.lp, p_l)

    device = typeof(RE) <: CuArray ? :GPU : :CPU
    Rout = typeof(RE) <: CuArray ? CUDA.zeros(R, slcp, slb, slt) : zeros(R, slcp, slb, slt)

    A = reshape(A, (slt, srt, slct, srct))
    B = reshape(B, (slb, srb, slcb, srcb))

    if srcb * slct >= srct * slcb
        pls = SparseCSC(R, M.lp, p_lt, p_l, p_lb, device)
        prs = SparseCSC(R, M.lp, p_rt, p_r, p_rb, device)
        B2 = permutedims(B, (3, 1, 4, 2))  # [lcb, lb, rcb, rb]
        B2 = reshape(B2, (slcb * slb, srcb * srb))  # [(lcb, lb), (rcb, rb)]
        A2 = permutedims(A, (3, 4, 1, 2))

        Rtemp1 = typeof(RE) <: CuArray ? CuArray{R}(undef, (slct, slc * slcb * slb)) : Array{R}(undef, (slct, slc * slcb * slb))
        Rtemp2 = typeof(RE) <: CuArray ? CuArray{R}(undef, (slcp, slb)) : Array{R}(undef, (slcp, slb))
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
                mul!(Rtemp1, (@view A2[:, :, ilt, irt]), Rslc)
                mul!(Rtemp2, pls', reshape(Rtemp1, (slct * slc * slcb, slb)))
                Rout[:, :, ilt] += Rtemp2 # [rcp, rb]
                # Rtemp = (@view A2[:, :, ilt, irt]) * Rslc  # [lct, (lc, lcb, lb)]
                # Rtemp = reshape(Rtemp, (slct * slc * slcb, slb))
                # Rout[:, :, ilt] += pls' * Rtemp  # [lcp, lb]
            end
        end
    else
        pls = SparseCSC(R, M.lp, p_lb, p_l, p_lt, device)
        prs = SparseCSC(R, M.lp, p_rb, p_r, p_rt, device)
        A2 = permutedims(A, (3, 1, 4, 2))  # [lcb, lb, rcb, rb]
        A2 = reshape(A2, (slct * slt, srct * srt))  # [(lct, lt), (rct, rt)]
        B2 = permutedims(B, (3, 4, 1, 2))
        Rtemp1 = typeof(RE) <: CuArray ? CuArray{R}(undef, (slcb, slc * slct * slt)) : Array{R}(undef, (slcb, slc * slct * slt))
        Rtemp2 = typeof(RE) <: CuArray ? CuArray{R}(undef, (slcp, slt)) : Array{R}(undef, (slcp, slt))
        for irb ∈ 1 : srb
            Rslc = RE[irb, :, :]  # [rt, rc]
            Rslc = prs * Rslc'  # [(rcb, rc, rct), rt]
            Rslc = reshape(Rslc, (srcb * src, srct * srt))  # [(rcb, rc), (rct, rt)]
            Rslc = Rslc * A2'  # [(rcb, rc), (rct, rt)] * [(lct, lt), (rct, rt)]'
            Rslc = reshape(Rslc, (srcb, src, slct * slt))  # [rcb, rc, (lct, lt)]
            Rslc = permutedims(Rslc, (1, 3, 2))  # [rcb, (lct, lt), rc]
            Rslc = contract_matrix_tensor3(M.con, Rslc)  # [rcb, (lct, lt), lc]
            Rslc = permutedims(Rslc, (1, 3, 2))  # [rcb, lc, (lct, lt)]
            Rslc = reshape(Rslc, (srcb, slc * slct * slt))  # [rcb, (lc, lct, lt)]
            for ilb ∈ 1 : slb
                mul!(Rtemp1, (@view B2[:, :, ilb, irb]), Rslc)
                mul!(Rtemp2, pls', reshape(Rtemp1, (slcb * slc * slct, slt)))
                Rout[:, ilb, :] += Rtemp2 # [lcp, lt]
                # Rtemp = (@view B2[:, :, ilb, irb]) * Rslc  # [lcb, (lc, lct, lt)]
                # Rtemp = reshape(Rtemp, (slcb * slc * slct, slt))
                # Rout[:, ilb, :] += pls' * Rtemp  # [lcp, lt]
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
    device = typeof(RE) <: CuArray ? :GPU : :CPU

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
