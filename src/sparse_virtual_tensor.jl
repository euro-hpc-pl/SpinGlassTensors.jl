# """
# $(TYPEDSIGNATURES)
# """
# function update_env_left(
#     LE::S, A::S, M::T, B::S, ::Val{:n}
# ) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}
#     h = M.con
#     p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

#     A = CUDA.CuArray(A)
#     B = CUDA.CuArray(B)
#     LE = CUDA.CuArray(LE)

#     @cast A4[al, ab1, ab2, ar] := A[al, (ab1, ab2), ar] (ab1 ∈ 1:maximum(p_lt))
#     @cast B4[bl, bt1, bt2, br] := B[bl, (bt1, bt2), br] (bt1 ∈ 1:maximum(p_lb))

#     slb = size(LE, 1) #lb, lc, lt
#     LEn = permutedims(LE, (2, 1, 3))
#     @cast LEn[lc, (lb, lt)] := LEn[lc, lb, lt]
#     tLE = typeof(LE)
#     ps = projectors_to_sparse(p_lb, p_l, p_lt, tLE)
#     Ltemp = ps * LEn

#     @cast Ltemp[lcb, lc, lct, lb, lt] := Ltemp[(lcb, lc, lct), (lb, lt)] (lcb ∈ 1:maximum(p_lb), lc ∈ 1:maximum(p_l), lb ∈ 1:slb)
#     Ltemp = permutedims(Ltemp, (4, 1, 2, 5, 3))

#     # sh = size(h)
#     # sA4 = size(A4)
#     # sB4 = size(B4)

#     # if sh[2] / sh[1] < sA4[3] * sA4[4] / (sA4[1] * sA4[2]) < sB4[3] * sB4[4] / (sB4[1] * sB4[2])
#     #     @cast Ltemp[(lc, lc2), lt, (lc1, lb)] := Ltemp[lc, lc2, lt, lc1, lb]
#     #     Ltemp = attach_central_left(Ltemp, h, Val(:n))
#     #     @cast Ltemp[lc, lc2, lt, lc1, lb] := Ltemp[(lc, lc2), lt, (lc1, lb)] (lc2 ∈ 1:maximum(p_lb), lb ∈ 1:maximum(p_lt))
#     #     @tensor Ltemp[bl, bt1, c, ar, ab2] := Ltemp[bl, bt1, c, al, ab1] * A4[al, ab1, ab2, ar]
#     #     @tensor Ltemp[br, bt2, c, ar, ab2] := Ltemp[bl, bt1, c, ar, ab2] * B4[bl, bt1, bt2, br]
#     # elseif sh[2] / sh[1] < sB4[3] * sB4[4] / (sB4[1] * sB4[2]) < sA4[3] * sA4[4] / (sA4[1] * sA4[2])
#     #     @cast Ltemp[(lc, lc2), lt, (lc1, lb)] := Ltemp[lc, lc2, lt, lc1, lb]
#     #     Ltemp = attach_central_left(Ltemp, h, Val(:n))
#     #     @cast Ltemp[lc, lc2, lt, lc1, lb] := Ltemp[(lc, lc2), lt, (lc1, lb)] (lc2 ∈ 1:maximum(p_lb), lb ∈ 1:maximum(p_lt))
#     #     @tensor Ltemp[br, bt2, c, al, ab1] := Ltemp[bl, bt1, c, al, ab1] * B4[bl, bt1, bt2, br]
#     #     @tensor Ltemp[br, bt2, c, ar, ab2] := Ltemp[br, bt2, c, al, ab1] * A4[al, ab1, ab2, ar]
#     # elseif sA4[3] * sA4[4] / (sA4[1] * sA4[2]) < sh[2] / sh[1] < sB4[3] * sB4[4] / (sB4[1] * sB4[2])
#     #     @tensor Ltemp[bl, bt1, c, ar, ab2] := Ltemp[bl, bt1, c, al, ab1] * A4[al, ab1, ab2, ar]
#     #     @cast Ltemp[(lc, lc2), lt, (lc1, lb)] := Ltemp[lc, lc2, lt, lc1, lb]
#     #     Ltemp = attach_central_left(Ltemp, h, Val(:n))
#     #     @cast Ltemp[lc, lc2, lt, lc1, lb] := Ltemp[(lc, lc2), lt, (lc1, lb)] (lc2 ∈ 1:maximum(p_lb), lb ∈ 1:maximum(p_lt))
#     #     @tensor Ltemp[br, bt2, c, ar, ab2] := Ltemp[bl, bt1, c, ar, ab2] * B4[bl, bt1, bt2, br]
#     # elseif sB4[3] * sB4[4] / (sB4[1] * sB4[2]) <  sh[2] / sh[1] < sA4[3] * sA4[4] / (sA4[1] * sA4[2])
#     #     @tensor Ltemp[br, bt2, c, al, ab1] := Ltemp[bl, bt1, c, al, ab1] * B4[bl, bt1, bt2, br]
#     #     @cast Ltemp[(lc, lc2), lt, (lc1, lb)] := Ltemp[lc, lc2, lt, lc1, lb]
#     #     Ltemp = attach_central_left(Ltemp, h, Val(:n))
#     #     @cast Ltemp[lc, lc2, lt, lc1, lb] := Ltemp[(lc, lc2), lt, (lc1, lb)] (lc2 ∈ 1:maximum(p_lb), lb ∈ 1:maximum(p_lt))
#     #     @tensor Ltemp[br, bt2, c, ar, ab2] := Ltemp[br, bt2, c, al, ab1] * A4[al, ab1, ab2, ar]
#     # elseif sA4[3] * sA4[4] / (sA4[1] * sA4[2]) < sB4[3] * sB4[4] / (sB4[1] * sB4[2]) < sh[2] / sh[1] 
#     #     @tensor Ltemp[bl, bt1, c, ar, ab2] := Ltemp[bl, bt1, c, al, ab1] * A4[al, ab1, ab2, ar]
#     #     @tensor Ltemp[br, bt2, c, ar, ab2] := Ltemp[bl, bt1, c, ar, ab2] * B4[bl, bt1, bt2, br]
#     #     @cast Ltemp[(lc, lc2), lt, (lc1, lb)] := Ltemp[lc, lc2, lt, lc1, lb]
#     #     Ltemp = attach_central_left(Ltemp, h, Val(:n))
#     #     @cast Ltemp[lc, lc2, lt, lc1, lb] := Ltemp[(lc, lc2), lt, (lc1, lb)] (lc2 ∈ 1:maximum(p_lb), lb ∈ 1:maximum(p_lt))
#     # elseif sB4[3] * sB4[4] / (sB4[1] * sB4[2]) < sA4[3] * sA4[4] / (sA4[1] * sA4[2]) < sh[2] / sh[1]
#     #     @tensor Ltemp[br, bt2, c, al, ab1] := Ltemp[bl, bt1, c, al, ab1] * B4[bl, bt1, bt2, br]
#     #     @tensor Ltemp[br, bt2, c, ar, ab2] := Ltemp[br, bt2, c, al, ab1] * A4[al, ab1, ab2, ar]
#     #     @cast Ltemp[(lc, lc2), lt, (lc1, lb)] := Ltemp[lc, lc2, lt, lc1, lb]
#     #     Ltemp = attach_central_left(Ltemp, h, Val(:n))
#     #     @cast Ltemp[lc, lc2, lt, lc1, lb] := Ltemp[(lc, lc2), lt, (lc1, lb)] (lc2 ∈ 1:maximum(p_lb), lb ∈ 1:maximum(p_lt))
#     # end

#     @cast Ltemp[(lc, lct), lt, (lcb, lb)] := Ltemp[lc, lct, lt, lcb, lb]
#     Ltemp = attach_central_left(Ltemp, h, Val(:n))
#     @cast Ltemp[lc, lct, lt, lcb, lb] := Ltemp[(lc, lct), lt, (lcb, lb)] (lct ∈ 1:maximum(p_lb), lb ∈ 1:maximum(p_lt))
#     @tensor Ltemp[nlc, nlct, lt, nlcb, nlb] := Ltemp[lc, lct, lt, lcb, lb] * A4[lcb, lb, nlb, nlcb] * B4[lc, lct, nlct, nlc] order = (lc, lct, lcb, lb)

#     slb = size(Ltemp, 1)
#     prs = projectors_to_sparse_transposed(p_rb, p_r, p_rt, tLE) 
#     Ltemp = permutedims(Ltemp, (2, 3, 5, 1, 4)) #[(lcb, lc, lct), (lb, lt)]
#     @cast Ltemp[(lcb, lc, lct), (lb, lt)] := Ltemp[lcb, lc, lct, lb, lt] 
#     Lnew = prs * Ltemp  #[cc, (nb, nt)]
#     @cast Lnew[lc, lb, lt] := Lnew[lc, (lb, lt)] (lb ∈ 1:slb)
#     Array(permutedims(Lnew, (2, 1, 3)) ./ maximum(abs.(Lnew)))
# end

function update_env_left(
    LE::S, A::S, M::T, B::S, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    A = CUDA.CuArray(A)
    B = CUDA.CuArray(B)
    LE = CUDA.CuArray(LE)

    @cast A4[al, ab1, ab2, ar] := A[al, (ab1, ab2), ar] (ab1 ∈ 1:maximum(p_lt))
    @cast B4[bl, bt1, bt2, br] := B[bl, (bt1, bt2), br] (bt1 ∈ 1:maximum(p_lb))

    slb = size(LE, 1) #lb, lc, lt
    LEn = permutedims(LE, (2, 1, 3))
    @cast LEn[lc, (lb, lt)] := LEn[lc, lb, lt]
    tLE = typeof(LE)
    ps = projectors_to_sparse(p_lb, p_l, p_lt, tLE)
    Ltemp = ps * LEn

    @cast Ltemp[lcb, lc, lct, lb, lt] := Ltemp[(lcb, lc, lct), (lb, lt)] (lcb ∈ 1:maximum(p_lb), lc ∈ 1:maximum(p_l), lb ∈ 1:slb)
    Ltemp = permutedims(Ltemp, (4, 1, 2, 5, 3))
    @cast Ltemp[(lb, lcb), lc, (lt, lct)] := Ltemp[lb, lcb, lc, lt, lct]
    Ltemp = attach_central_left(Ltemp, h)
    @cast Ltemp[lb, lcb, lc, lt, lct] := Ltemp[(lb, lcb), lc, (lt, lct)] (lcb ∈ 1:maximum(p_lb), lct ∈ 1:maximum(p_lt))
    #@tensor Ltemp[nlc, nlct, lt, nlcb, nlb] := Ltemp[lc, lct, lt, lcb, lb] * A4[lcb, lb, nlb, nlcb] * B4[lc, lct, nlct, nlc] order = (lc, lct, lcb, lb)
    @tensor Ltemp[nlcb, nlb, lc, nlct, nlt] := Ltemp[lb, lcb, lc, lt, lct] * A4[lt, lct, nlt, nlct] * B4[lb, lcb, nlb, nlcb] order = (lb, lcb, lt, lct)

    slb = size(Ltemp, 1)
    prs = projectors_to_sparse_transposed(p_rb, p_r, p_rt, tLE) 
    Ltemp = permutedims(Ltemp, (2, 3, 5, 1, 4)) #[(lcb, lc, lct), (lb, lt)]
    @cast Ltemp[(nlcb, lc, nlct), (nlb, nlt)] := Ltemp[nlcb, lc, nlct, nlb, nlt] 
    Lnew = prs * Ltemp  #[cc, (nb, nt)]
    @cast Lnew[lc, lb, lt] := Lnew[lc, (lb, lt)] (lb ∈ 1:slb)
    Array(permutedims(Lnew, (2, 1, 3)) ./ maximum(abs.(Lnew)))
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_left(
    LE::S, A::S, M::T, B::S, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    A = CUDA.CuArray(A)
    B = CUDA.CuArray(B)
    LE = CUDA.CuArray(LE)

    @cast A4[al, ab1, ab2, ar] := A[al, (ab1, ab2), ar] (ab1 ∈ 1:maximum(p_lb))
    @cast B4[bl, bt1, bt2, br] := B[bl, (bt1, bt2), br] (bt1 ∈ 1:maximum(p_lt))

    slb = size(LE, 1)
    LEn = permutedims(LE, (2, 1, 3))
    @cast LEn[lc, (lb, lt)] := LEn[lc, lb, lt]
    tLE = typeof(LE)
    ps = projectors_to_sparse(p_lb, p_l, p_lt, tLE)
    Ltemp = ps * LEn

    @cast Ltemp[lc1, lc, lc2, lb, lt] := Ltemp[(lc1, lc, lc2), (lb, lt)] (lc1 ∈ 1:maximum(p_lb), lc ∈ 1:maximum(p_l), lb ∈ 1:slb)
    Ltemp = permutedims(Ltemp, (4, 1, 2, 5, 3))

    @cast Ltemp[(lc, lc2), lt, (lc1, lb)] := Ltemp[lc, lc2, lt, lc1, lb]
    Ltemp = attach_central_left(Ltemp, h)
    @cast Ltemp[lc, lc2, lt, lc1, lb] := Ltemp[(lc, lc2), lt, (lc1, lb)] (lc2 ∈ 1:maximum(p_lb), lb ∈ 1:maximum(p_lt))
    @tensor Ltempnew[br, ab2, c, ar, bt2] := Ltemp[bl, ab1, c, al, bt1] * A4[al, ab1, ab2, ar] * B4[bl, bt1, bt2, br] order = (bl, bt1, al, ab1)

    slb = size(Ltempnew, 1)
    prs = projectors_to_sparse_transposed(p_rb, p_r, p_rt, tLE) 

    Ltempnew = permutedims(Ltempnew, (5, 3, 2, 1, 4)) #[(nbp, nc, ntp), (nb, nt)]
    @cast Ltempnew[(ar, c, bt2), (ab2, br)] := Ltempnew[ar, c, bt2, ab2, br] 

    Lnew = prs * Ltempnew  #[cc, (nb, nt)]

    @cast Lnew[c, nb, nt] := Lnew[c, (nb, nt)] (nb ∈ 1:slb)
    Array(permutedims(Lnew, (2, 1, 3)) ./ maximum(abs.(Lnew)))
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_right(
    RE::S, A::S, M::T, B::S, ::Val{:n}
) where {T <: SparseVirtualTensor, S <: AbstractArray{Float64,3}}

    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    A = CUDA.CuArray(A)
    B = CUDA.CuArray(B)
    RE = CUDA.CuArray(RE)

    @cast A4[al, ab1, ab2, ar] := A[al, (ab1, ab2), ar] (ab1 ∈ 1:maximum(p_lt))
    @cast B4[bl, bt1, bt2, br] := B[bl, (bt1, bt2), br] (bt1 ∈ 1:maximum(p_lb))

    srt = size(RE, 1)
    REn = permutedims(RE, (2, 3, 1))
    @cast REn[rc, (rb, rt)] := REn[rc, rb, rt]
    tRE = typeof(RE)
    ps = projectors_to_sparse(p_rb, p_r, p_rt, tRE)
    Rtemp = ps * REn

    @cast Rtemp[rc1, rc, rc2, rb, rt] := Rtemp[(rc1, rc, rc2), (rb, rt)] (rc1 ∈ 1:maximum(p_rb), rc ∈ 1:maximum(p_r), rt ∈ 1:srt)
    Rtemp = permutedims(Rtemp, (5, 3, 2, 4, 1))
    @cast Rtemp[(rt, rc2), rc, (rb, rc1)] :=  Rtemp[rt, rc2, rc, rb, rc1]
    Rtemp = attach_central_right(Rtemp, h)
    @cast Rtemp[rt, rc2, rc, rb, rc1] := Rtemp[(rt, rc2), rc, (rb, rc1)] (rc1 ∈ 1:maximum(p_rb), rc2 ∈ 1:maximum(p_rt))
    @tensor Rtempnew[al, ab1, c, bl, bt1] := Rtemp[ar, ab2, c, br, bt2] * A4[al, ab1, ab2, ar] * B4[bl, bt1, bt2, br] #order = (b, tp, t, bp)

    srt = size(Rtempnew, 1)
    prs = projectors_to_sparse_transposed(p_lb, p_l, p_lt, tRE) 
    Rtempnew = permutedims(Rtempnew, (5, 3, 2, 1, 4)) #[(nbp, nc, ntp), (nb, nt)]
    @cast Rtempnew[(bl, c, ab1), (bt1, al)] :=  Rtempnew[bl, c, ab1, bt1, al] 
    Rnew = prs * Rtempnew  #[cc, (nb, nt)] 
    @cast Rnew[c, nt, nb] := Rnew[c, (nt, nb)] (nt ∈ 1:srt)
    Array(permutedims(Rnew, (2, 1, 3)) ./ maximum(abs.(Rnew)))
end


"""
$(TYPEDSIGNATURES)
"""
function update_env_right(
    RE::S, A::S, M::T, B::S, ::Val{:c}
) where {T <: SparseVirtualTensor, S <: AbstractArray{Float64,3}}
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    A = CUDA.CuArray(A)
    B = CUDA.CuArray(B)
    RE = CUDA.CuArray(RE)

    @cast A4[al, ab1, ab2, ar] := A[al, (ab1, ab2), ar] (ab1 ∈ 1:maximum(p_lb))
    @cast B4[bl, bt1, bt2, br] := B[bl, (bt1, bt2), br] (bt1 ∈ 1:maximum(p_lt))

    srt = size(RE, 1)
    REn = permutedims(RE, (2, 3, 1))
    @cast REn[rc, (rb, rt)] := REn[rc, rb, rt]
    tRE = typeof(RE)
    ps = projectors_to_sparse(p_rb, p_r, p_rt, tRE)
    Rtemp = ps * REn

    @cast Rtemp[rc1, rc, rc2, rb, rt] := Rtemp[(rc1, rc, rc2), (rb, rt)] (rc1 ∈ 1:maximum(p_rb), rc ∈ 1:maximum(p_r), rt ∈ 1:srt)
    Rtemp = permutedims(Rtemp, (5, 3, 2, 4, 1))

    @cast Rtemp[(rt, rc2), rc, (rb, rc1)] :=  Rtemp[rt, rc2, rc, rb, rc1]
    Rtemp = attach_central_right(Rtemp, h)
    @cast Rtemp[rt, rc2, rc, rb, rc1] := Rtemp[(rt, rc2), rc, (rb, rc1)] (rc1 ∈ 1:maximum(p_rb), rc2 ∈ 1:maximum(p_rt))
    @tensor Rtempnew[al, bt1, c, bl, ab1] := Rtemp[ar, bt2, c, br, ab2] * A4[al, ab1, ab2, ar] * B4[bl, bt1, bt2, br] #order = (b, tp, t, bp)

    srt = size(Rtempnew, 1)
    prs = projectors_to_sparse_transposed(p_lb, p_l, p_lt, tRE) 

    Rtempnew = permutedims(Rtempnew, (2, 3, 5, 1, 4)) #[(nbp, nc, ntp), (nb, nt)]
    @cast Rtempnew[(bl, al, bt1), (ab1, c)] :=  Rtempnew[bl, al, bt1, ab1, c] 
    Rnew = prs * Rtempnew  #[cc, (nb, nt)] 

    @cast Rnew[c, nt, nb] := Rnew[c, (nt, nb)] (nt ∈ 1:srt)
    Array(permutedims(Rnew, (2, 1, 3)) ./ maximum(abs.(Rnew)))
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
