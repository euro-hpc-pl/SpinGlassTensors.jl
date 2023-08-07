# virtual.jl: contractions with VirtualTensor on CPU and CUDA
export update_env_left2

# @memoize Dict
alloc_undef(R, onGPU, shape) = onGPU ? CuArray{R}(undef, shape) : Array{R}(undef, shape)
alloc_zeros(R, onGPU, shape) = onGPU ? CUDA.zeros(R, shape) : zeros(R, shape)

function proj_out(lp, k1, k2, k3, device)
    p1 = get_projector!(lp, k1, device)
    p2 = get_projector!(lp, k2, device)
    p3 = get_projector!(lp, k3, device)
    @assert length(p1) == length(p2) == length(p3)
    s1, s2 = size(lp, k1), size(lp, k2)
    p1 .+ s1 * (p2 .- 1) .+ s1 * s2 * (p3 .- 1)
end

function proj_2step_12(lp, (k1, k2), k3, device)
    p1 = get_projector!(lp, k1, :CPU)
    p2 = get_projector!(lp, k2, :CPU)
    p3 = get_projector!(lp, k3, device)
    @assert length(p1) == length(p2) == length(p3)
    s1 = size(lp, k1)

    p12, transitions_matrix = rank_reveal(hcat(p1, p2), :PE)
    (p1, p2) = Tuple(Array(t) for t ∈ eachcol(transitions_matrix))

    s12 = maximum(p12)

    if device == :CPU
        p12 = CuArray(p12)
        p1  = CuArray(p1)
        p2  = CuArray(p2)
    end

    pf1 = p12 .+ s12 .* (p3 .- 1)
    pf2 = p1 .+ s1 .* (p2 .- 1)

    pf1, pf2, s12
end

function proj_2step_23(lp, k1, (k2, k3), device)
    p1 = get_projector!(lp, k1, device)
    p2 = get_projector!(lp, k2, :CPU)
    p3 = get_projector!(lp, k3, :CPU)
    @assert length(p1) == length(p2) == length(p3)

    s1, s2 = size(lp, k1), size(lp, k2)

    p23, transitions_matrix = rank_reveal(hcat(p2, p3), :PE)
    (p2, p3) = Tuple(Array(t) for t ∈ eachcol(transitions_matrix))

    s23 = maximum(p23)

    if device == :CPU
        p23 = CuArray(p23)
        p2  = CuArray(p2)
        p3  = CuArray(p3)
    end

    pf1 = p1 .+ s1 .* (p23 .- 1)
    pf2 = p2 .+ s2 .* (p3 .- 1)

    pf1, pf2, s23
end

function update_env_left(LE::S, A::S, M::VirtualTensor{R, 4}, B::S) where {S <: Tensor{R, 3}} where R <: Real
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs
    slb, srb = size(B, 1), size(B, 2)
    slt, srt = size(A, 1), size(A, 2)
    slcb, slc, slct = size(M.lp, p_lb), size(M.lp, p_l), size(M.lp, p_lt)
    srcb, src, srct = size(M.lp, p_rb), size(M.lp, p_r), size(M.lp, p_rt)
    srcp = length(M.lp, p_r)

    onGPU = typeof(LE) <: CuArray
    device = onGPU ? :GPU : :CPU

    A = reshape(A, (slt, srt, slct, srct))
    B = reshape(B, (slb, srb, slcb, srcb))
    Lout = alloc_zeros(R, onGPU, (srb, srt, srcp))
    if slcb * srct >= slct * srcb
        pls = proj_out(M.lp, p_lb, p_lt, p_l, device)
        prs = proj_out(M.lp, p_rb, p_r, p_rt, device)
        # prs = SparseCSC(R, M.lp, p_rb, p_r, p_rt, device)
        B2 = permutedims(B, (1, 3, 2, 4))  # [lb, lcb, rb, rcb]
        B2 = reshape(B2, (slb * slcb, srb * srcb))  # [(lb, lcb), (rb, rcb)]
        tmp1 = alloc_zeros(R, onGPU, (slb, slcb * slct * slc))
        tmp2 = alloc_undef(R, onGPU, (srb * srcb, slct * slc))
        tmp5 = alloc_undef(R, onGPU, (srb * srcb, src, slct))
        tmp7 = alloc_undef(R, onGPU, (srb * srcb * src, srct))
        tmp8 = alloc_undef(R, onGPU, (srcp, srb))
        @assert length(pls) == length(Set(Array(pls)))
        @assert length(prs) == length(Set(Array(prs)))
        for ilt ∈ 1 : slt
            tmp1[:, pls] = (@view LE[:, ilt, :])  # [lb, (lcb, lct, lc)]
            mul!(tmp2, B2', reshape(tmp1, (slb * slcb, slct * slc)))  # [(lb, lcb), (rb, rcb)]' * [(lb, lcb), (lct, lc)]
            tmp3 = reshape(tmp2, (srb * srcb, slct, slc))  # [(rb, rcb), lct, lc]
            tmp4 = contract_tensor3_matrix(tmp3, M.con)  # [(rb, rcb), lct, rc]
            permutedims!(tmp5, tmp4, (1, 3, 2))  # [(rb, rcb), rc, lct]
            tmp6 = reshape(tmp5, (srb * srcb * src, slct))  # [(rb, rcb, rc), lct]
            for irt ∈ 1 : srt
                mul!(tmp7, tmp6, (@view A[ilt, irt, :, :]))
                # mul!(tmp8, prs', reshape(tmp7, (srb, srcb * src * srct))')
                # Lout[:, irt, :] .+= tmp8'  # [rb, rcp]
                tmp8 = reshape(tmp7, (srb, srcb * src * srct))
                Lout[:, irt, :] .+= tmp8[:, prs]  # [rb, rcp]
            end
        end
    else
        pls = proj_out(M.lp, p_lt, p_lb, p_l, device)
        # prs = SparseCSC(R, M.lp, p_rt, p_r, p_rb, device)
        prs = proj_out(M.lp, p_rt, p_r, p_rb, device)
        A2 = permutedims(A, (1, 3, 2, 4))  # [lt, lct, rt, rct]
        A2 = reshape(A2, (slt * slct, srt * srct))  # [(lt, lct), (rt, rct)]
        tmp1 = alloc_zeros(R, onGPU, (slt, slct * slcb * slc))
        tmp2 = alloc_undef(R, onGPU, (srt * srct, slcb * slc))
        tmp5 = alloc_undef(R, onGPU, (srt * srct, src, slcb))
        tmp7 = alloc_undef(R, onGPU, (srt * srct * src, srcb))
        tmp8 = alloc_undef(R, onGPU, (srcp, srt))
        @assert length(pls) == length(Set(Array(pls)))
        @assert length(prs) == length(Set(Array(prs)))
        for ilb ∈ 1 : slb
            tmp1[:, pls] = (@view LE[ilb, :, :])  # [lt, (lct, lcb, lc)]
            mul!(tmp2, A2', reshape(tmp1, (slt * slct, slcb * slc)))
            tmp3 = reshape(tmp2, (srt * srct, slcb, slc))  # [(rt, rct), lcb, lc]
            tmp4 = contract_tensor3_matrix(tmp3, M.con)  # [(rt, rct), lcb, rc]
            permutedims!(tmp5, tmp4, (1, 3, 2))  # [(rt, rct), rc, lcb]
            tmp6 = reshape(tmp5, (srt * srct * src, slcb))  # [(rt, rct, rc), lcb]
            for irb ∈ 1 : srb
                mul!(tmp7, tmp6, (@view B[ilb, irb, :, :]))
                # mul!(tmp8, prs', reshape(tmp7, (srt, srct * src * srcb))')
                # Lout[irb, :, :] .+= tmp8'  # [rt, rcp]
                tmp8 = reshape(tmp7, (srt, srct * src * srcb))
                Lout[irb, :, :] .+= tmp8[:, prs]  # [rt, rcp]
            end
        end
    end
    Lout  # [rb, rt, rcp]
end




function update_env_left2(LE::S, A::S, M::VirtualTensor{R, 4}, B::S) where {S <: Tensor{R, 3}} where R <: Real
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs
    slb, srb = size(B, 1), size(B, 2)
    slt, srt = size(A, 1), size(A, 2)
    slcb, slc, slct = size(M.lp, p_lb), size(M.lp, p_l), size(M.lp, p_lt)
    srcb, src, srct = size(M.lp, p_rb), size(M.lp, p_r), size(M.lp, p_rt)
    srcp = length(M.lp, p_r)

    onGPU = typeof(LE) <: CuArray
    device = onGPU ? :GPU : :CPU

    A = reshape(A, (slt, srt, slct, srct))
    B = reshape(B, (slb, srb, slcb, srcb))
    Lout = alloc_zeros(R, onGPU, (srb, srt, srcp))

    s_lb = size(M.lp, p_lb)
    s_lt = size(M.lp, p_lt)
    p_lb = get_projector!(M.lp, p_lb, device)
    p_lt = get_projector!(M.lp, p_lt, :CPU)
    p_l = get_projector!(M.lp, p_l, :CPU)
    p_llt, transitions_matrix = rank_reveal(hcat(p_lt, p_l), :PE)
    (p_lt, p_l) = Tuple(Array(t) for t ∈ eachcol(transitions_matrix))
    scllt = maximum(p_llt)

    if onGPU
        p_llt = CuArray(p_llt)
        p_lt = CuArray(p_lt)
        p_l = CuArray(p_l)
    end
    pls1 = p_lb .+ s_lb .* (p_llt .- 1)
    pls2 = p_lt .+ s_lt .* (p_l .- 1)

    s_rb = size(M.lp, p_rb)
    p_rt = get_projector!(M.lp, p_rt, device)
    p_rb = get_projector!(M.lp, p_rb, :CPU)
    p_r = get_projector!(M.lp, p_r, :CPU)
    p_rrb, transitions_matrix = rank_reveal(hcat(p_rb, p_r), :PE)
    (p_rb, p_r) = Tuple(Array(t) for t ∈ eachcol(transitions_matrix))
    scrrb = maximum(p_rrb)
    if onGPU
        p_rrb = CuArray(p_rrb)
        p_rb = CuArray(p_rb)
        p_r = CuArray(p_r)
    end
    prs1 = p_rrb.+ scrrb .* (p_rt .- 1)
    prs2 = p_rb .+ s_rb .* (p_r .- 1)
    # prs = SparseCSC(R, M.lp, p_rb, p_r, p_rt, device)

    B2 = permutedims(B, (1, 3, 2, 4))  # [lb, lcb, rb, rcb]
    B2 = reshape(B2, (slb * slcb, srb * srcb))  # [(lb, lcb), (rb, rcb)]

    @assert length(pls1) == length(Set(Array(pls1)))
    @assert length(pls2) == length(Set(Array(pls2)))
    @assert length(prs1) == length(Set(Array(prs1)))
    @assert length(prs2) == length(Set(Array(prs2)))

    tmp1 = alloc_zeros(R, onGPU, (slb, slcb * scllt))
    tmp3 = alloc_zeros(R, onGPU, (srb * srcb, slct * slc))

    for ilt ∈ 1 : slt
        tmp1[:, pls1] = (@view LE[:, ilt, :])  # [lb, (lcb, lct, lc)]
        tmp2 = B2' * reshape(tmp1, (slb * slcb, scllt))  # [(lb, lcb), (rb, rcb)]' * [(lb, lcb), scllt] = [(rb, rcb), scllt]
        tmp3[:, pls2] = tmp2  # [(rb, rcb), (lct, lc)]
        tmp4 = reshape(tmp3, (srb * srcb, slct, slc))  # [(rb, rcb), lct, lc]
        tmp5 = contract_tensor3_matrix(tmp4, M.con)  # [(rb, rcb), lct, rc]
        tmp6 = permutedims(tmp5, (1, 3, 2))  # [(rb, rcb), rc, lct]
        tmp7 = reshape(tmp6, (srb, srcb * src, slct))  # [rb, (rcb, rc), lct]
        tmp8 = tmp7[:, prs2, :]  # [rb, crrb, lct]
        for irt ∈ 1 : srt
            tmp9 =  reshape(tmp8, (srb * scrrb, slct)) * (@view A[ilt, irt, :, :])
            tmp10 = reshape(tmp9, (srb, scrrb * srct))
            Lout[:, irt, :] .+= tmp10[:, prs1]  # [rb, rcp]
        end
    end
    Lout  # [rb, rt, rcp]
end


function project_ket_on_bra(LE::S, B::S, M::VirtualTensor{R, 4}, RE::S) where {S <: Tensor{R, 3}} where R <: Real
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs
    slb, slt = size(LE, 1), size(LE, 2)
    srb, srt = size(RE, 1), size(RE, 2)
    slcb, slc, slct = size(M.lp, p_lb), size(M.lp, p_l), size(M.lp, p_lt)
    srcb, src, srct = size(M.lp, p_rb), size(M.lp, p_r), size(M.lp, p_rt)

    onGPU = typeof(LE) <: CuArray
    device = onGPU ? :GPU : :CPU

    B = reshape(B, (slb, srb, slcb, srcb))
    B2 = permutedims(B, (1, 3, 2, 4))  # [lb, lcb, rb, rcb]
    B2 = reshape(B2, (slb * slcb, srb * srcb))  # [(lb, lcb), (rb, rcb)]
    LR = alloc_zeros(R, onGPU, (slt, srt, slct, srct))

    if slcb >= srcb
        pls = proj_out(M.lp, p_lb, p_lt, p_l, device)
        prs = proj_out(M.lp, p_rb, p_r, p_rt, device)
        tmp1 = alloc_zeros(R, onGPU, (slb, slcb * slct * slc))
        tmp2 = alloc_undef(R, onGPU, (srb * srcb, slct * slc))
        tmp5 = alloc_undef(R, onGPU, (srcb * srb, src, slct))
        tmp7 = alloc_zeros(R, onGPU, (srb, srcb * src * srct))
        tmp8 = alloc_undef(R, onGPU, (slct, srct))
        @assert length(pls) == length(Set(Array(pls)))
        @assert length(prs) == length(Set(Array(prs)))
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
        @assert length(pls) == length(Set(Array(pls)))
        @assert length(prs) == length(Set(Array(prs)))
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
    reshape(LR, (slt, srt, slct * srct))
end


function update_env_right(RE::S, A::S, M::VirtualTensor{R, 4}, B::S) where {S <: Tensor{R, 3}} where R <: Real
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs
    slb, srb = size(B, 1), size(B, 2)
    slt, srt = size(A, 1), size(A, 2)
    slcb, slc, slct = size(M.lp, p_lb), size(M.lp, p_l), size(M.lp, p_lt)
    srcb, src, srct = size(M.lp, p_rb), size(M.lp, p_r), size(M.lp, p_rt)
    slcp = length(M.lp, p_l)

    onGPU = typeof(RE) <: CuArray
    device = onGPU ? :GPU : :CPU

    A = reshape(A, (slt, srt, slct, srct))
    B = reshape(B, (slb, srb, slcb, srcb))
    Rout = alloc_zeros(R, onGPU, (slb, slt, slcp))

    if srcb * slct >= srct * slcb
        pls = proj_out(M.lp, p_lb, p_l, p_lt, device)
        #pls = SparseCSC(R, M.lp, p_lb, p_l, p_lt, device)
        prs = proj_out(M.lp, p_rb, p_rt, p_r, device)
        B2 = permutedims(B, (1, 3, 2, 4))  # [lb, lcb, rb, rcb]
        B2 = reshape(B2, (slb * slcb, srb * srcb))  # [(lb, lcb), (rb, rcb)]
        tmp1 = alloc_zeros(R, onGPU, (srb, srcb * srct * src))
        tmp2 = alloc_undef(R, onGPU, (slb * slcb, srct * src))
        tmp5 = alloc_undef(R, onGPU, (slb * slcb, slc, srct))
        tmp7 = alloc_undef(R, onGPU, (slb * slcb * slc, slct))
        tmp8 = alloc_undef(R, onGPU, (slcp, slb))
        @assert length(pls) == length(Set(Array(pls)))
        @assert length(prs) == length(Set(Array(prs)))
        for irt ∈ 1 : srt
            tmp1[:, prs] = (@view RE[:, irt, :])  # [rb, (rcb, rct, rc)]
            mul!(tmp2, B2, reshape(tmp1, (srb * srcb, srct * src)))  # [(lb, lcb), (rb, rcb)] * [(rb, rcb), (rct, rc)]
            tmp3 = reshape(tmp2, (slb * slcb, srct, src))  # [(lb, lcb), rct, rc]
            tmp4 = contract_matrix_tensor3(M.con, tmp3)  # [(lb, lcb), rct, lc]
            permutedims!(tmp5, tmp4, (1, 3, 2))  # [(lb, lcb), lc, rct]
            tmp6 = reshape(tmp5, (slb * slcb * slc, srct))  # [(lb, lcb, lc), rct]
            for ilt ∈ 1 : slt
                mul!(tmp7, tmp6, (@view A[ilt, irt, :, :])')
                # mul!(tmp8, pls', reshape(tmp7, (slb, slcb * slc * slct))')
                # Rout[:, ilt, :] .+= tmp8'
                tmp8 = reshape(tmp7, (slb, slcb * slc * slct))
                Rout[:, ilt, :] .+= tmp8[:, pls]
            end
        end
    else
        pls = proj_out(M.lp, p_lt, p_l, p_lb, device)
        #pls = SparseCSC(R, M.lp, p_lt, p_l, p_lb, device)
        prs = proj_out(M.lp, p_rt, p_rb, p_r, device)
        A2 = permutedims(A, (1, 3, 2, 4))  # [lt, lct, rt, rct]
        A2 = reshape(A2, (slt * slct, srt * srct))  # [(lt, lct), (rt, rct)]
        tmp1 = alloc_zeros(R, onGPU, (srt, srct * srcb * src))
        tmp2 = alloc_undef(R, onGPU, (slt * slct, srcb * src))
        tmp5 = alloc_undef(R, onGPU, (slt * slct, slc, srcb))
        tmp7 = alloc_undef(R, onGPU, (slt * slct * slc, slcb))
        tmp8 = alloc_undef(R, onGPU, (slcp, slt))
        @assert length(pls) == length(Set(Array(pls)))
        @assert length(prs) == length(Set(Array(prs)))
        for irb ∈ 1 : srb
            tmp1[:, prs] = (@view RE[irb, :, :])  # [rt, (rct, rcb, rc)]
            mul!(tmp2, A2, reshape(tmp1, (srt * srct, srcb * src)))
            tmp3 = reshape(tmp2, (slt * slct, srcb, src))  # [(lt, lct), rcb, rc]
            tmp4 = contract_matrix_tensor3(M.con, tmp3)  # [(lt, lct), rcb, lc]
            permutedims!(tmp5, tmp4, (1, 3, 2))  # [(lt, lct), lc, rcb]
            tmp6 = reshape(tmp5, (slt * slct * slc, srcb))  # [(lt, lct, lc), rcb]
            for ilb ∈ 1 : slb
                mul!(tmp7, tmp6, (@view B[ilb, irb, :, :])')
                # mul!(tmp8, pls', reshape(tmp7, (slt, slct * slc * slcb))')
                # Rout[ilb, :, :] .+= tmp8'
                tmp8 = reshape(tmp7, (slt, slct * slc * slcb))
                Rout[ilb, :, :] .+= tmp8[:, pls]
            end
        end
    end
    Rout
end




# function update_reduced_env_right(
#     K::Tensor{R, 1}, RE::Tensor{R, 2}, M::VirtualTensor{R, 4}, B::Tensor{R, 3}
# ) where R <: Real
#     p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

#     slb, srb = size(B, 1), size(B, 2)
#     slcb, slc, slct = size(M.lp, p_lb), size(M.lp, p_l), size(M.lp, p_lt)
#     srcb, src, srct = size(M.lp, p_rb), size(M.lp, p_r), size(M.lp, p_rt)

#     onGPU = typeof(RE) <: CuArray
#     device = onGPU ? :GPU : :CPU

#     K = reshape(K, (slct, srct))  # [lct, rct]
#     B = reshape(B, (slb, srb, slcb, srcb))  # [lb, rb, lcb, rcb]
#     B2 = permutedims(B, (3, 1, 4, 2))  # [lcb, lb, rcb, rb]
#     B2 = reshape(B2, (slcb * slb, srcb * srb))  # [(lcb, lb), (rcb, rb)]

#     pls = SparseCSC(R, M.lp, p_lt, p_l, p_lb, device)
#     prs = SparseCSC(R, M.lp, p_rt, p_r, p_rb, device)
#     Rtemp = prs * RE'  # [(rct, rc, rcb), rb]
#     if srcb * slct >= srct * slcb
#         Rtemp = reshape(Rtemp, (srct * src, srcb * srb))  # [(rct, rc), (rcb, rb)]
#         Rtemp = Rtemp * B2'  # [(rct, rc), (rcb, rb)] * [(lcb, lb), (rcb, rb)]'
#         Rtemp = reshape(Rtemp, (srct, src, slcb * slb))  # [rct, rc, (lcb, lb)]
#         Rtemp = permutedims(Rtemp, (1, 3, 2))  # [rct, (lcb, lb), rc]
#         Rtemp = contract_matrix_tensor3(M.con, Rtemp)  # [rct, (lcb, lb), lc]
#         Rtemp = permutedims(Rtemp, (1, 3, 2))  # [rct, lc, (lcb, lb)]
#         Rtemp = reshape(Rtemp, (srct, slc * slcb * slb))  # [rct, (lc, lcb, lb)]
#         Rtemp = K * Rtemp  # [lct, (lc, lcb, lb)]
#     else
#         Rtemp = reshape(Rtemp, (srct, src * srcb * srb))  # [rct, (rc, rcb, rb)]
#         Rtemp = K * Rtemp  # [lct, (rc, rcb, rb)]
#         Rtemp = reshape(Rtemp, (slct, src, srcb * srb))  # [lct, rc, (rcb, rb)]
#         Rtemp = permutedims(Rtemp, (1, 3, 2))  # [lct, (rcb, rb), rc]
#         Rtemp = contract_matrix_tensor3(M.con, Rtemp)  # [lct, (rcb, rb), lc]
#         Rtemp = permutedims(Rtemp, (1, 3, 2))  # [lct, lc, (rcb, rb)]
#         Rtemp = reshape(Rtemp, (slct * slc, srcb * srb))  # [(lct, lc), (rcb, rb)]
#         Rtemp = Rtemp * B2'  # [(lct, lc), (lcb, lb)]
#     end
#     Rtemp = reshape(Rtemp, (slct * slc * slcb, slb))
#     Rtemp = pls' * Rtemp  # [lcp, lb]
#     Rtemp = permutedims(Rtemp, (2, 1))  # [lb, lcp]
#     Rtemp
# end


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
    B2 = permutedims(B, (1, 3, 2, 4))  # [lb, lcb, rb, rcb]
    B2 = reshape(B2, (slb * slcb, srb * srcb))  # [(lb, lcb), (rb, rcb)]


    if srcb * slct >= srct * slcb
        prs = proj_out(M.lp, p_rb, p_rt, p_r, device)
        pls = proj_out(M.lp, p_lb, p_l, p_lt, device)
        Rtemp = alloc_zeros(R, onGPU, (srb, srcb * srct * src))
        Rtemp[:, prs] = RE  # [rb, (rcb, rct, rc)]
        Rtemp = reshape(Rtemp, (srb * srcb, srct * src))  # [(rb, rcb), (rct, rc)]
        Rtemp = B2 * Rtemp
        Rtemp = reshape(Rtemp, (slb * slcb, srct, src))  # [(lb, lcb), rct, rc]
        Rtemp = contract_matrix_tensor3(M.con, Rtemp)  # [(lb, lcb), rct, lc]
        Rtemp = permutedims(Rtemp, (1, 3, 2))  # [(lb, lcb), lc, rct]
        Rtemp = reshape(Rtemp, (slb * slcb * slc, srct))  # [(lb, lcb, lc), rct]
        Rtemp = Rtemp * K' # [(lb, lcb, lc), lct]
        Rtemp = reshape(Rtemp, (slb, slcb * slc * slct))
        Rtemp = Rtemp[:, pls]
    else
        prs = proj_out(M.lp, p_rb, p_r, p_rt, device)
        pls = proj_out(M.lp, p_lb, p_lt, p_l, device)
        Rtemp = alloc_zeros(R, onGPU, (srb, srcb * src * srct))
        Rtemp[:, prs] = RE  # [rb, (rcb, rct, rc)]
        Rtemp = reshape(Rtemp, (srb * srcb * src, srct))  # [(rb, rcb, rc), rct]
        Rtemp = Rtemp * K' # [(rb, rcb, rc), lct]
        Rtemp = reshape(Rtemp, (srb * srcb, src, slct))  # [(rb, rcb), rc, lct]
        Rtemp = permutedims(Rtemp, (1, 3, 2))  # [(rb, rcb), lct, rc]
        Rtemp = contract_matrix_tensor3(M.con, Rtemp)  # [(rb, rcb), lct, lc]
        Rtemp = reshape(Rtemp, (srb * srcb, slct * slc))  # [(rb, rcb), (lct, lc)]
        Rtemp = B2 * Rtemp  # [(lb, lcb), (lct, lc)]
        Rtemp = reshape(Rtemp, (slb, slcb * slct * slc))
        Rtemp = Rtemp[:, pls]
    end
    Rtemp
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

function corner_matrix(C::S, M::T, B::S) where {S <: Tensor{R, 3}, T <: VirtualTensor{R, 4}} where R <: Real
    slb, srb = size(B, 1), size(B, 2)
    srcc, stc = size(C, 2), size(C, 3)
    V = contract_tensors43(M, B)
    vl, vr, vt = size(V, 1), size(V, 2), size(V, 3)
    V = reshape(V, (vl, srb, stc, vt))
    @tensor Cnew[vl, vt, vrr] := V[vl, srb, stc, vt] * C[srb, vrr, stc]
    reshape(Cnew, (slb, :, srcc, vt))
end