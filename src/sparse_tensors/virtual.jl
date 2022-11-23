function CUDA.CUSPARSE.CuSparseMatrixCSC(
    ::Type{T}, p_lb::R, p_l::R, p_lt::R
) where {T <: Real, R <: Vector{Int}}
    @assert length(p_lb) == length(p_l) == length(p_lt)

    p_l, p_lb, p_lt = CuArray.((p_l, p_lb, p_lt))
    ncol = length(p_lb)
    n = maximum(p_l)
    m = maximum(p_lb)

    CuSparseMatrixCSC(
        CuArray(1:ncol+1),
        n * m * (p_lt .- 1) .+ m * (p_l .- 1) .+ p_lb,
        CUDA.ones(T, ncol),
        (n * m * maximum(p_lt), ncol)
    )
end

function CUDA.CUSPARSE.CuSparseMatrixCSR(
    ::Type{T}, p_lb::R, p_l::R, p_lt::R
) where {T <: Real, R <: Vector{Int}}
    transpose(CuSparseMatrixCSC(T, p_lb, p_l, p_lt))
end

@inline r2_over_r1(A) = size(A, 2) / size(A, 1)
@inline r1_over_r2(A) = 1 / r2_over_r1(A)

"""
Select optimal order of attaching matrices to L
"""
function attach_3_matrices_left(L, B2, h, A2)
    if r2_over_r1(h) <= r2_over_r1(B2) <= r2_over_r1(A2)
        L = attach_central_left(L, h)  # [..., rc, ...] = [..., lc, ...] * [lc, rc]
        @tensor L[rfb, x, y] := L[lfb, x, y] * B2[lfb, rfb]
        @tensor L[x, y, rft] := L[x, y, lft] * A2[lft, rft]
    elseif r2_over_r1(h) <= r2_over_r1(A2) <= r2_over_r1(B2)
        L = attach_central_left(L, h)  # [..., rc, ...] = [..., lc, ...] * [lc, rc]
        @tensor L[x, y, rft] := L[x, y, lft] * A2[lft, rft]
        @tensor L[rfb, x, y] := L[lfb, x, y] * B2[lfb, rfb]
    elseif r2_over_r1(A2) <= r2_over_r1(h) <= r2_over_r1(B2)
        @tensor L[x, y, rft] := L[x, y, lft] * A2[lft, rft]
        L = attach_central_left(L, h)  # [..., rc, ...] = [..., lc, ...] * [lc, rc]
        @tensor L[rfb, x, y] := L[lfb, x, y] * B2[lfb, rfb]
    elseif r2_over_r1(A2) <= r2_over_r1(B2) <= r2_over_r1(h)
        @tensor L[x, y, rft] := L[x, y, lft] * A2[lft, rft]
        @tensor L[rfb, x, y] := L[lfb, x, y] * B2[lfb, rfb]
        L = attach_central_left(L, h)  # [..., rc, ...] = [..., lc, ...] * [lc, rc]
    elseif r2_over_r1(B2) <= r2_over_r1(h) <= r2_over_r1(A2)
        @tensor L[rfb, x, y] := L[lfb, x, y] * B2[lfb, rfb]
        L = attach_central_left(L, h)  # [..., rc, ...] = [..., lc, ...] * [lc, rc]
        @tensor L[x, y, rft] := L[x, y, lft] * A2[lft, rft]
    else # r2_over_r1(B2) <= r2_over_r1(A2) <= r2_over_r1(h)
        @tensor L[rfb, x, y] := L[lfb, x, y] * B2[lfb, rfb]
        @tensor L[x, y, rft] := L[x, y, lft] * A2[lft, rft]
        L = attach_central_left(L, h)  # [..., rc, ...] = [..., lc, ...] * [lc, rc]
    end
    L
end

"""
Select optimal order of attaching matrices to R
"""
function attach_3_matrices_right(R, B2, h, A2)
    if r1_over_r2(h) <= r1_over_r2(B2) <= r1_over_r2(A2)
        R = attach_central_right(R, h)  # [..., lc, ...] = [..., rc, ...] * [lc, rc]
        @tensor R[lfb, x, y] := R[rfb, x, y] * B2[lfb, rfb]
        @tensor R[x, y, lft] := R[x, y, rft] * A2[lft, rft]
    elseif r1_over_r2(h) <= r1_over_r2(A2) <= r1_over_r2(B2)
        R = attach_central_right(R, h)  # [..., lc, ...] = [..., rc, ...] * [lc, rc]
        @tensor R[x, y, lft] := R[x, y, rft] * A2[lft, rft]
        @tensor R[lfb, x, y] := R[rfb, x, y] * B2[lfb, rfb]
    elseif r1_over_r2(A2) <= r1_over_r2(h) <= r1_over_r2(B2)
        @tensor R[x, y, lft] := R[x, y, rft] * A2[lft, rft]
        R = attach_central_right(R, h)  # [..., lc, ...] = [..., rc, ...] * [lc, rc]
        @tensor R[lfb, x, y] := R[rfb, x, y] * B2[lfb, rfb]
    elseif r1_over_r2(A2) <= r1_over_r2(B2) <= r1_over_r2(h)
        @tensor R[x, y, lft] := R[x, y, rft] * A2[lft, rft]
        @tensor R[lfb, x, y] := R[rfb, x, y] * B2[lfb, rfb]
        R = attach_central_right(R, h)  # [..., lc, ...] = [..., rc, ...] * [lc, rc]
    elseif r1_over_r2(B2) <= r1_over_r2(h) <= r1_over_r2(A2)
        @tensor R[lfb, x, y] := R[rfb, x, y] * B2[lfb, rfb]
        R = attach_central_right(R, h)  # [..., lc, ...] = [..., rc, ...] * [lc, rc]
        @tensor R[x, y, lft] := R[x, y, rft] * A2[lft, rft]
    else # r1_over_r2(B2) <= r1_over_r2(A2) <= r1_over_r2(h)
        @tensor R[lfb, x, y] := R[rfb, x, y] * B2[lfb, rfb]
        @tensor R[x, y, lft] := R[x, y, rft] * A2[lft, rft]
        R = attach_central_right(R, h)  # [..., lc, ...] = [..., rc, ...] * [lc, rc]
    end
    R
end

function attach_2_matrices(L, B2, h, R)
    h1, h2 = size(h, 1), size(h, 2)
    b1, b2 = size(B2, 1), size(B2, 2)
    leg_list = [h1, h2, b1, b2]
    _, max_index = findmax(leg_list)
    biggest_leg = leg_list[max_index]

    if biggest_leg ∈ [h1, h2]
        if h1 >= h2
            L = attach_central_left(L, h)
        else
            R = attach_central_right(R, h)
        end
        if  b1 >= b2
            @tensor L[rfb, x, y] := L[lfb, x, y] * B2[lfb, rfb]
        else
            @tensor R[x, y, lfb] := R[x, y, rfb] * B2[lfb, rfb]
        end
    else
        if  b1 >= b2
            @tensor L[rfb, x, y] := L[lfb, x, y] * B2[lfb, rfb]
        else
            @tensor R[x, y, lfb] := R[x, y, rfb] * B2[lfb, rfb]
        end
        if h1 >= h2
            L = attach_central_left(L, h)
        else
            R = attach_central_right(R, h)
        end
    end
    @tensor LR[lft, rft] := L[lfb, lfh, lft] * R[rft, lfh, lfb]
    LR
end


function update_env_left(
    LE::S, A::S, M::SparseVirtualTensor, B::S, ::Val{:n}
) where S <: ArrayOrCuArray{3}
    A, B, L = CuArray.((A, B, LE))

    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    slb, srb = size(B, 1), size(B, 3)
    slt, srt = size(A, 1), size(A, 3)
    slcb, slc, slct = maximum(p_lb), maximum(p_l), maximum(p_lt)
    srcb, srct = maximum(p_rb), maximum(p_rt)
    srcp = length(p_r)

    batch_size = 1

    F = eltype(LE)
    Lout = CUDA.zeros(F, srcp, srb, srt)

    @cast A2[(lt, lct), (rct, rt)] := A[lt, (lct, rct), rt] (lct ∈ 1:slct)
    ps = CuSparseMatrixCSC(eltype(LE), p_lb, p_l, p_lt)
    prs = CuSparseMatrixCSR(eltype(LE), p_rb, p_r, p_rt)

    lb_from = 1
    while lb_from <= slb
        lb_to = min(lb_from + batch_size - 1, slb)

        Lslc = L[lb_from : lb_to, :, :]
        Lslc = permutedims(Lslc, (2, 1, 3))  # [lcp, lb, lt]
        @cast Lslc[lcp, (lb, lt)] := Lslc[lcp, lb, lt]
        Lslc = ps * Lslc #[(lcb, lc, lct), (lb, lt)]

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


function update_env_left(
    LE::S, A::S, M::SparseVirtualTensor, B::S, ::Val{:c}
) where S <: ArrayOrCuArray{3}
    A, B, L = CuArray.((A, B, LE))

    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    slb, srb = size(B, 1), size(B, 3)
    slt, srt = size(A, 1), size(A, 3)
    slcb, slc, slct = maximum(p_lb), maximum(p_l), maximum(p_lt)
    srcb, srct = maximum(p_rb), maximum(p_rt)
    srcp = length(p_r)

    batch_size = 1

    F = eltype(LE)
    Lout = CUDA.zeros(F, srcp, srb, srt)

    @cast A2[(lt, lcb), (rcb, rt)] := A[lt, (lcb, rcb), rt] (lcb ∈ 1:slcb)

    ps = CuSparseMatrixCSC(eltype(LE), p_lb, p_l, p_lt)
    prs = CuSparseMatrixCSR(eltype(LE), p_rb, p_r, p_rt)

    lb_from = 1
    while lb_from <= slb
        lb_to = min(lb_from + batch_size - 1, slb)

        Lslc = L[lb_from : lb_to, :, :]
        Lslc = permutedims(Lslc, (2, 1, 3))  # [lcp, lb, lt]
        @cast Lslc[lcp, (lb, lt)] := Lslc[lcp, lb, lt]
        Lslc = ps * Lslc #[(lcb, lc, lct), (lb, lt)]

        @cast Lslc[lcb, lc, lct, lb, lt] := Lslc[(lcb, lc, lct), (lb, lt)] (lcb ∈ 1:slcb, lc ∈ 1:slc, lt ∈ 1:slt)
        Lslc = permutedims(Lslc, (4, 3, 2, 5, 1)) #[lb, lcb, lc, lt, lct]
        @cast Lslc[(lb, lct), lc, (lt, lcb)] := Lslc[lb, lct, lc, lt, lcb]

        rb_from = 1
        while rb_from <= srb
            rb_to = min(rb_from + batch_size - 1, srb)

            Btemp = B[lb_from : lb_to, :, rb_from : rb_to]
            @cast B2[(lb, lct), (rct, rb)] := Btemp[lb, (lct, rct), rb] (lct ∈ 1:slct)

            Ltemp = attach_3_matrices_left(Lslc, B2, h, A2)
            
            @cast Ltemp[rct, rb, rc, rcb, rt] := Ltemp[(rct, rb), rc, (rcb, rt)] (rct ∈ 1:srct, rcb ∈ 1:srcb)

            Ltemp = permutedims(Ltemp, (1, 3, 4, 2, 5))  # [rcb, rc, rct, rb, rt]
            @cast Ltemp[(rct, rc, rcb), (rb, rt)] := Ltemp[rct, rc, rcb, rb, rt]

            Ltemp = prs * Ltemp  # [rcp, (rb, rt)]
            @cast Ltemp[rcp, rb, rt] := Ltemp[rcp, (rb, rt)] (rt ∈ 1:srt)

            Lout[:, rb_from : rb_to, :] += Ltemp

            rb_from = rb_to + 1
        end
        lb_from = lb_to + 1
    end

    Array(permutedims(Lout, (2, 1, 3)) ./ maximum(abs.(Lout))) #[rb, rcp, rt]
end

function update_env_right(
    RE::S, A::S, M::SparseVirtualTensor, B::S, ::Val{:n}
) where S <: ArrayOrCuArray{3}
    A, B, R = CuArray.((A, B, RE))

    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    slb, srb = size(B, 1), size(B, 3)
    slt, srt = size(A, 1), size(A, 3)
    slcb, slc, slct = maximum(p_lb), maximum(p_l), maximum(p_lt)
    srcb, src, srct = maximum(p_rb), maximum(p_r), maximum(p_rt)
    slcp = length(p_l)

    batch_size = 1

    F = eltype(RE)
    Rout = CUDA.zeros(F, slcp, slt, slb)

    @cast A2[(lt, lct), (rct, rt)] := A[lt, (lct, rct), rt] (rct ∈ 1:srct)
    prs = CuSparseMatrixCSR(eltype(RE), p_lb, p_l, p_lt)
    ps = CuSparseMatrixCSC(eltype(RE), p_rb, p_r, p_rt)

    rb_from = 1
    while rb_from <= srb
        rb_to = min(rb_from + batch_size - 1, srb)

        Rslc = R[:, :, rb_from : rb_to]
        Rslc = permutedims(Rslc, (2, 3, 1))  # [rcp, rb, rt]
        @cast Rslc[rcp, (rb, rt)] := Rslc[rcp, rb, rt]
        Rslc = ps * Rslc #[(rcb, rc, rct), (rb, rt)]
        @cast Rslc[rcb, rc, rct, rb, rt] := Rslc[(rcb, rc, rct), (rb, rt)] (rcb ∈ 1:srcb, rc ∈ 1:src, rt ∈ 1:srt)
        Rslc = permutedims(Rslc, (1, 4, 2, 3, 5)) #[rcb, rb, rc, rct, rt]
        @cast Rslc[(rcb, rb), rc, (rct, rt)] := Rslc[rcb, rb, rc, rct, rt]

        lb_from = 1
        while lb_from <= slb
            lb_to = min(lb_from + batch_size - 1, slb)

            Btemp = B[lb_from : lb_to, :, rb_from : rb_to]
            @cast B2[(lb, lcb), (rcb, rb)] := Btemp[lb, (lcb, rcb), rb] (rcb ∈ 1:srcb)

            Rtemp = attach_3_matrices_right(Rslc, B2, h, A2)

            @cast Rtemp[lb, lcb, lc, lt, lct] := Rtemp[(lb, lcb), lc, (lt, lct)] (lcb ∈ 1:slcb, lt ∈ 1:slt)

            Rtemp = permutedims(Rtemp, (2, 3, 5, 4, 1)) #[lcb, lc, lct, lb, lt]
            @cast Rtemp[(lcb, lc, lct), (lt, lb)] := Rtemp[lcb, lc, lct, lt, lb]
            Rtemp = prs * Rtemp  # [lcp, (lb, lt)]
            @cast Rtemp[lcp, lt, lb] := Rtemp[lcp, (lt, lb)] (lt ∈ 1:slt)
            Rout[:, :, lb_from : lb_to] += Rtemp

            lb_from = lb_to + 1
        end
        rb_from = rb_to + 1
    end
    Array(permutedims(Rout, (2, 1, 3)) ./ maximum(abs.(Rout))) #[lb, lcp, lt]
end


function update_env_right(
    RE::S, A::S, M::SparseVirtualTensor, B::S, ::Val{:c}
) where S <: ArrayOrCuArray{3}
    A, B, R = CuArray.((A, B, RE))

    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    slb, srb = size(B, 1), size(B, 3)
    slt, srt = size(A, 1), size(A, 3)
    slcb, slc, slct = maximum(p_lb), maximum(p_l), maximum(p_lt)
    srcb, src, srct = maximum(p_rb), maximum(p_r), maximum(p_rt)
    slcp = length(p_l)

    batch_size = 1

    F = eltype(RE)
    Rout = CUDA.zeros(F, slcp, slt, slb)

    @cast A2[(lt, lcb), (rcb, rt)] := A[lt, (lcb, rcb), rt] (rcb ∈ 1:srcb)
    prs = CuSparseMatrixCSR(eltype(RE), p_lb, p_l, p_lt)
    ps = CuSparseMatrixCSC(eltype(RE), p_rb, p_r, p_rt)

    rb_from = 1
    while rb_from <= srb
        rb_to = min(rb_from + batch_size - 1, srb)

        Rslc = R[:, :, rb_from : rb_to]
        Rslc = permutedims(Rslc, (2, 3, 1))  # [rcp, rb, rt]
        @cast Rslc[rcp, (rb, rt)] := Rslc[rcp, rb, rt]

        Rslc = ps * Rslc #[(rcb, rc, rct), (rb, rt)]

        @cast Rslc[rcb, rc, rct, rb, rt] := Rslc[(rcb, rc, rct), (rb, rt)] (rcb ∈ 1:srcb, rc ∈ 1:src, rt ∈ 1:srt)
        Rslc = permutedims(Rslc, (3, 4, 2, 1, 5)) #[rct, rb, rc, rcb, rt]
        @cast Rslc[(rct, rb), rc, (rcb, rt)] := Rslc[rct, rb, rc, rcb, rt]

        lb_from = 1
        while lb_from <= slb
            lb_to = min(lb_from + batch_size - 1, slb)

            Btemp = B[lb_from : lb_to, :, rb_from : rb_to]
            @cast B2[(lb, lct), (rct, rb)] := Btemp[lb, (lct, rct), rb] (rct ∈ 1:srct)

            Rtemp = attach_3_matrices_right(Rslc, B2, h, A2)

            @cast Rtemp[lb, lct, lc, lt, lcb] := Rtemp[(lb, lct), lc, (lt, lcb)] (lct ∈ 1:slct, lt ∈ 1:slt)

            Rtemp = permutedims(Rtemp, (5, 3, 2, 4, 1)) #[lcb, lc, lct, lb, lt]
            @cast Rtemp[(lcb, lc, lct), (lt, lb)] := Rtemp[lcb, lc, lct, lt, lb]
            Rtemp = prs * Rtemp  # [lcp, (lb, lt)]
            @cast Rtemp[lcp, lt, lb] := Rtemp[lcp, (lt, lb)] (lt ∈ 1:slt)
            Rout[:, :, lb_from : lb_to] += Rtemp

            lb_from = lb_to + 1
        end
        rb_from = rb_to + 1
    end
    Array(permutedims(Rout, (2, 1, 3)) ./ maximum(abs.(Rout))) #[lb, lcp, lt]
end

# function project_ket_on_bra(
#     LE::S, B::S, M::SparseVirtualTensor, RE::S, ::Val{:n}
# ) where S <: ArrayOrCuArray{3}
#     h = M.con
#     p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

#     B, L, R = CuArray.((B, LE, RE))
#     F = eltype(LE)

#     sl1, sl3 = size(L, 1), size(L, 3)
#     sr1, sr3 = size(R, 1), size(R, 3)
#     slcb, slc, slct = maximum(p_lb), maximum(p_l), maximum(p_lt)
#     srcb, src, srct = maximum(p_rb), maximum(p_r), maximum(p_rt)

#     batch_size = 1

#     ps = CuSparseMatrixCSC(eltype(LE), p_lb, p_l, p_lt)
#     prs = CuSparseMatrixCSC(eltype(RE), p_rb, p_r, p_rt)

#     @cast B2[(lb, lcb), (rcb, rb)] := B[lb, (lcb, rcb), rb] (rcb ∈ 1:srcb)
#     Lout = CUDA.zeros(F, size(L, 1), slcb, slc, size(L, 3), slct) #[lb, lcb, lc, lt, lct]
#     Rout = CUDA.zeros(F, srct, size(R, 1), src, srcb, size(R, 3)) #[rct, rt, rc, rcb, rb]

#     l_from = 1
#     while l_from <= sl3
#         l_to = min(l_from + batch_size - 1, sl3)
#         Lslc = L[:, :, l_from : l_to]

#         Lslc = permutedims(Lslc, (2, 1, 3))  # [lcp, lb, lt]
#         @cast Lslc[lcp, (lb, lt)] := Lslc[lcp, lb, lt]

#         Lslc = ps * Lslc #[(lcb, lc, lct), (lb, lt)]

#         @cast Lslc[lcb, lc, lct, lb, lt] := Lslc[(lcb, lc, lct), (lb, lt)] (lcb ∈ 1:slcb, lc ∈ 1:slc, lb ∈ 1:sl1)
#         Lslc = permutedims(Lslc, (4, 1, 2, 5, 3)) #[lb, lcb, lc, lt, lct]
#         Lout[:, :, :, l_from:l_to, :] += Lslc #Lout[lb, lcb, lc, lt, lct]
        
#         l_from = l_to + 1
#     end
        
#     @cast Lout[(lb, lcb), lc, (lt, lct)] := Lout[lb, lcb, lc, lt, lct]

#     r_from = 1
#     while r_from <= sr1
#         r_to = min(r_from + batch_size - 1, sr1)
    
#         Rslc = R[r_from : r_to, :, :]
#         Rslc = permutedims(Rslc, (2, 3, 1))  # [rcp, rb, rt]
#         @cast Rslc[rcp, (rb, rt)] := Rslc[rcp, rb, rt]
#         Rslc = prs * Rslc #[(rcb, rc, rct), (rb, rt)]
    
#         @cast Rslc[rcb, rc, rct, rb, rt] := Rslc[(rcb, rc, rct), (rb, rt)] (rcb ∈ 1:srcb, rc ∈ 1:src, rb ∈ 1:sr3)
#         Rslc = permutedims(Rslc, (3, 5, 2, 1, 4)) #[rct, rt, rc, rcb, rb]

#         Rout[:, r_from:r_to, :, :, :] += Rslc

#         r_from = r_to + 1
#     end

#     @cast Rout[(rct, rt), rc, (rcb, rb)] := Rout[rct, rt, rc, rcb, rb]

#     LR = attach_2_matrices(Lout, B2, h, Rout)
#     @cast LR[lt, (lct, rct), rt] := LR[(lt, lct), (rct, rt)] (lct ∈ 1:slct, rct ∈ 1:srct)

#     Array(LR ./ maximum(abs.(LR)))
# end

function project_ket_on_bra(
    LE::S, B::S, M::SparseVirtualTensor, RE::S, ::Val{:n}
) where S <: ArrayOrCuArray{3}
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    B, L, R = CuArray.((B, LE, RE))
    F = eltype(LE)

    sl1, sl3 = size(L, 1), size(L, 3)
    sr1, sr3 = size(R, 1), size(R, 3)
    slcb, slc, slct = maximum(p_lb), maximum(p_l), maximum(p_lt)
    srcb, src, srct = maximum(p_rb), maximum(p_r), maximum(p_rt)

    batch_size = 1

    ps = CuSparseMatrixCSC(eltype(LE), p_lb, p_l, p_lt)
    prs = CuSparseMatrixCSC(eltype(RE), p_rb, p_r, p_rt)

    @cast B2[(lb, lcb), (rcb, rb)] := B[lb, (lcb, rcb), rb] (rcb ∈ 1:srcb)
    
    LRout = CUDA.zeros(F, sl3, slct * srct, sr1)
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


function project_ket_on_bra(
    LE::S, B::S, M::SparseVirtualTensor, RE::S, ::Val{:c}
) where S <: ArrayOrCuArray{3}
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    B, L, R = CuArray.((B, LE, RE))
    F = eltype(LE)

    sl1, sl3 = size(L, 1), size(L, 3)
    sr1, sr3 = size(R, 1), size(R, 3)
    slcb, slc, slct = maximum(p_lb), maximum(p_l), maximum(p_lt)
    srcb, src, srct = maximum(p_rb), maximum(p_r), maximum(p_rt)

    batch_size = 1

    ps = CuSparseMatrixCSC(eltype(LE), p_lt, p_l, p_lb)
    prs = CuSparseMatrixCSC(eltype(RE), p_rb, p_r, p_rt)

    @cast B2[(lb, lct), (rct, rb)] := B[lb, (lct, rct), rb] (rct ∈ 1:srct)
    
    LRout = CUDA.zeros(F, sl3, slcb * srcb, sr1)
    l_from = 1
    while l_from <= sl3
        l_to = min(l_from + batch_size - 1, sl3)
        Lslc = L[:, :, l_from : l_to]

        Lslc = permutedims(Lslc, (2, 1, 3))  # [lcp, lb, lt]
        @cast Lslc[lcp, (lb, lt)] := Lslc[lcp, lb, lt]

        Lslc = ps * Lslc #[(lcb, lc, lct), (lb, lt)]

        @cast Lslc[lct, lc, lcb, lb, lt] := Lslc[(lct, lc, lcb), (lb, lt)] (lcb ∈ 1:slcb, lc ∈ 1:slc, lb ∈ 1:sl1)
        Lslc = permutedims(Lslc, (4, 1, 2, 5, 3)) #[lb, lcb, lc, lt, lct]
        @cast Lslc[(lb, lct), lc, (lt, lcb)] := Lslc[lb, lct, lc, lt, lcb]

        r_from = 1
        while r_from <= sr1
            r_to = min(r_from + batch_size - 1, sr1)
        
            Rslc = R[r_from : r_to, :, :]
            Rslc = permutedims(Rslc, (2, 3, 1))  # [rcp, rb, rt]
            @cast Rslc[rcp, (rb, rt)] := Rslc[rcp, rb, rt]
            Rslc = prs * Rslc #[(rcb, rc, rct), (rb, rt)]
        
            @cast Rslc[rct, rc, rcb, rb, rt] := Rslc[(rct, rc, rcb), (rb, rt)] (rcb ∈ 1:srcb, rc ∈ 1:src, rb ∈ 1:sr3)
            Rslc = permutedims(Rslc, (3, 5, 2, 1, 4)) #[rct, rt, rc, rcb, rb]
            @cast Rslc[(rcb, rt), rc, (rct, rb)] := Rslc[rcb, rt, rc, rct, rb]
            LR = attach_2_matrices(Lslc, B2, h, Rslc)
            @cast LR[lt, (lcb, rcb), rt] := LR[(lt, lcb), (rcb, rt)] (lcb ∈ 1:slcb, rcb ∈ 1:srcb)
            LRout[l_from:l_to, :, r_from:r_to] += LR
            r_from = r_to + 1
        end
        l_from = l_to + 1
    end
        
    Array(LRout ./ maximum(abs.(LRout)))
end


# function project_ket_on_bra(
#     LE::S, B::S, M::SparseVirtualTensor, RE::S, ::Val{:c}
# ) where S <: ArrayOrCuArray{3}
#     h = M.con
#     p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

#     B, L, R = CuArray.((B, LE, RE))
#     F = eltype(LE)

#     sl1, sl3 = size(L, 1), size(L, 3)
#     sr1, sr3 = size(R, 1), size(R, 3)
#     slcb, slc, slct = maximum(p_lb), maximum(p_l), maximum(p_lt)
#     srcb, src, srct = maximum(p_rb), maximum(p_r), maximum(p_rt)

#     batch_size = 1

#     ps = CuSparseMatrixCSC(eltype(LE), p_lt, p_l, p_lb)
#     prs = CuSparseMatrixCSC(eltype(RE), p_rb, p_r, p_rt)

#     @cast B2[(lb, lct), (rct, rb)] := B[lb, (lct, rct), rb] (rct ∈ 1:srct)
#     Lout = CUDA.zeros(F, size(L, 1), slct, slc, size(L, 3), slcb) #[lb, lct, lc, lt, lcb]
#     Rout = CUDA.zeros(F, srcb, size(R, 1), src, srct, size(R, 3)) #[rcb, rt, rc, rct, rb]

#     l_from = 1
#     while l_from <= sl3
#         l_to = min(l_from + batch_size - 1, sl3)
#         Lslc = L[:, :, l_from : l_to]

#         Lslc = permutedims(Lslc, (2, 1, 3))  # [lcp, lb, lt]
#         @cast Lslc[lcp, (lb, lt)] := Lslc[lcp, lb, lt]

#         Lslc = ps * Lslc #[(lcb, lc, lct), (lb, lt)]

#         @cast Lslc[lct, lc, lcb, lb, lt] := Lslc[(lct, lc, lcb), (lb, lt)] (lcb ∈ 1:slcb, lc ∈ 1:slc, lb ∈ 1:sl1)
#         Lslc = permutedims(Lslc, (4, 1, 2, 5, 3)) #[lb, lct, lc, lt, lcb]
#         Lout[:, :, :, l_from:l_to, :] += Lslc #Lout[lb, lct, lc, lt, lcb]
        
#         l_from = l_to + 1
#     end
        
#     @cast Lout[(lb, lct), lc, (lt, lcb)] := Lout[lb, lct, lc, lt, lcb]

#     r_from = 1
#     while r_from <= sr1
#         r_to = min(r_from + batch_size - 1, sr1)
    
#         Rslc = R[r_from : r_to, :, :]
#         Rslc = permutedims(Rslc, (2, 3, 1))  # [rcp, rb, rt]
#         @cast Rslc[rcp, (rb, rt)] := Rslc[rcp, rb, rt]
#         Rslc = prs * Rslc #[(rcb, rc, rct), (rb, rt)]
    
#         @cast Rslc[rct, rc, rcb, rb, rt] := Rslc[(rct, rc, rcb), (rb, rt)] (rcb ∈ 1:srcb, rc ∈ 1:src, rb ∈ 1:sr3)
#         Rslc = permutedims(Rslc, (3, 5, 2, 1, 4)) #[rct, rt, rc, rcb, rb]

#         Rout[:, r_from:r_to, :, :, :] += Rslc

#         r_from = r_to + 1
#     end

#     @cast Rout[(rcb, rt), rc, (rct, rb)] := Rout[rcb, rt, rc, rct, rb]

#     LR = attach_2_matrices(Lout, B2, h, Rout)
#     @cast LR[lt, (lcb, rcb), rt] := LR[(lt, lcb), (rcb, rt)] (lcb ∈ 1:slcb, rcb ∈ 1:srcb)

#     Array(LR ./ maximum(abs.(LR)))
# end


# function project_ket_on_bra(
#     LE::S, B::S, M::SparseVirtualTensor, RE::S, ::Val{:n}
# ) where S <: ArrayOrCuArray{3}
#     h = M.con
#     p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

#     B, L, R = CuArray.((B, LE, RE))

#     slb, srb = size(B, 1), size(B, 3)
#     srcb, src, srct = maximum(p_rb), maximum(p_r), maximum(p_rt)
#     slcb, slc, slct = maximum(p_lb), maximum(p_l), maximum(p_lt)

#     @cast B2[(lb, lcb), (rcb, rb)] := B[lb, (lcb, rcb), rb] (lcb ∈ 1:slcb)

#     L = permutedims(L, (2, 1, 3))
#     @cast L[lc, (lb, lt)] := L[lc, lb, lt]
#     ps = CuSparseMatrixCSC(eltype(LE), p_lb, p_l, p_lt)
#     L = ps * L

#     @cast L[lcb, lc, lct, lb, lt] := L[(lcb, lc, lct), (lb, lt)] (lcb ∈ 1:slcb, lc ∈ 1:slc, lb ∈ 1:slb)
#     L = permutedims(L, (4, 1, 2, 5, 3))  # [lb, lcb, lc, lt, lct]
#     @cast L[(lb, lcb), lc, (lt, lct)] := L[lb, lcb, lc, lt, lct]

#     R = permutedims(R, (2, 3, 1))
#     @cast R[rc, (rb, rt)] := R[rc, rb, rt]
#     ps = CuSparseMatrixCSC(eltype(LE), p_rb, p_r, p_rt)
#     R = ps * R

#     @cast R[rcb, rc, rct, rb, rt] := R[(rcb, rc, rct), (rb, rt)] (rcb ∈ 1:srcb, rc ∈ 1:src, rb ∈ 1:srb)
#     R = permutedims(R, (3, 5, 2, 1, 4)) #[rct, rt, rc, rcb, rb]
#     @cast R[(rct, rt), rc, (rcb, rb)] := R[rct, rt, rc, rcb, rb]

#     LR = attach_2_matrices(L, B2, h, R)
#     @cast LR[lt, (lct, rct), rt] := LR[(lt, lct), (rct, rt)] (lct ∈ 1:slct, rct ∈ 1:srct)

#     Array(LR ./ maximum(abs.(LR)))
# end

# function project_ket_on_bra(
#     LE::S, B::S, M::SparseVirtualTensor, RE::S, ::Val{:c}
# ) where S <: ArrayOrCuArray{3}
#     h = M.con
#     p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

#     B, L, R = CuArray.((B, LE, RE))

#     slb, srb = size(B, 1), size(B, 3)
#     srcb, src, srct = maximum(p_rb), maximum(p_r), maximum(p_rt)
#     slcb, slc, slct = maximum(p_lb), maximum(p_l), maximum(p_lt)

#     @cast B2[(lb, lct), (rct, rb)] := B[lb, (lct, rct), rb] (lct ∈ 1:slct)

#     L = permutedims(L, (2, 1, 3))
#     @cast L[lc, (lb, lt)] := L[lc, lb, lt]
#     ps = CuSparseMatrixCSC(eltype(LE), p_lt, p_l, p_lb)
#     L = ps * L

#     @cast L[lct, lc, lcb, lb, lt] := L[(lct, lc, lcb), (lb, lt)] (lct ∈ 1:slct, lc ∈ 1:slc, lb ∈ 1:slb)
#     L = permutedims(L, (4, 1, 2, 5, 3))  # [lb, lct, lc, lt, lcb]

#     @cast L[(lb, lct), lc, (lt, lcb)] := L[lb, lct, lc, lt, lcb]

#     R = permutedims(R, (2, 3, 1))
#     @cast R[rc, (rb, rt)] := R[rc, rb, rt]
#     ps = CuSparseMatrixCSC(eltype(LE), p_rb, p_r, p_rt)
#     R = ps * R

#     @cast R[rct, rc, rcb, rb, rt] := R[(rct, rc, rcb), (rb, rt)] (rcb ∈ 1:srcb, rc ∈ 1:src, rb ∈ 1:srb)

#     R = permutedims(R, (3, 5, 2, 1, 4)) #[rct, rt, rc, rcb, rb]
#     @cast R[(rcb, rt), rc, (rct, rb)] := R[rcb, rt, rc, rct, rb]

#     LR = attach_2_matrices(L, B2, h, R)
#     @cast LR[lt, (lcb, rcb), rt] := LR[(lt, lcb), (rcb, rt)] (lcb ∈ 1:slcb, rcb ∈ 1:srcb)

#     Array(LR ./ maximum(abs.(LR)))
# end
