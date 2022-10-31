"""
$(TYPEDSIGNATURES)
"""
function update_env_left(
    LE::S, A::S, M::T, B::S, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    A = CUDA.CuArray(A)
    B = CUDA.CuArray(B)
    LE = CUDA.CuArray(LE)

    @cast A4[al, at, ar, ab] := A[al, (at, ar), ab] (at ∈ 1:maximum(p_lt))
    @cast B4[bl, bt, br, bb] := B[bl, (bt, br), bb] (bt ∈ 1:maximum(p_lb))

    slb = size(LE, 1) #lb, lc, lt
    LEn = permutedims(LE, (2, 1, 3))
    @cast LEn[lc, (lb, lt)] := LEn[lc, lb, lt]
    tLE = typeof(LE)
    ps = projectors_to_sparse(p_lb, p_l, p_lt, tLE)
    Ltemp = ps * LEn

    @cast Ltemp[nlb, lc, nlt, lb, lt] := Ltemp[(nlb, lc, nlt), (lb, lt)] (nlb ∈ 1:maximum(p_lb), lc ∈ 1:maximum(p_l), lb ∈ 1:slb)
    Ltemp = permutedims(Ltemp, (4, 1, 2, 5, 3))

    @cast Ltemp[(lc, nlt), lt, (nlb, lb)] := Ltemp[lc, nlt, lt, nlb, lb]
    Ltemp = attach_central_left(Ltemp, h, Val(:n))
    @cast Ltemp[lc, nlt, lt, nlb, lb] := Ltemp[(lc, nlt), lt, (nlb, lb)] (nlt ∈ 1:maximum(p_lb), lb ∈ 1:maximum(p_lt))
    @tensor Ltempnew[bb, br, c, ab, ar] := Ltemp[bl, bt, c, al, at] * A4[al, at, ar, ab] * B4[bl, bt, br, bb] order = (bl, bt, al, at)

    slb = size(Ltempnew, 1)
    prs = projectors_to_sparse_transposed(p_rb, p_r, p_rt, tLE) 

    Ltempnew = permutedims(Ltempnew, (2, 3, 5, 1, 4)) #[(nbp, nc, ntp), (nb, nt)]
    @cast Ltempnew[(br, c, ar), (bb, ab)] := Ltempnew[br, c, ar, bb, ab] 
    Lnew = prs * Ltempnew  #[cc, (nb, nt)]

    @cast Lnew[c, nb, nt] := Lnew[c, (nb, nt)] (nb ∈ 1:slb)
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

    @cast A4[al, at, ar, ab] := A[al, (at, ar), ab] (at ∈ 1:maximum(p_lb))
    @cast B4[bl, bt, br, bb] := B[bl, (bt, br), bb] (bt ∈ 1:maximum(p_lt))

    slb = size(LE, 1)
    LEn = permutedims(LE, (2, 1, 3))
    @cast LEn[lc, (lb, lt)] := LEn[lc, lb, lt]
    tLE = typeof(LE)
    ps = projectors_to_sparse(p_lb, p_l, p_lt, tLE)
    Ltemp = ps * LEn

    @cast Ltemp[nlb, lc, nlt, lb, lt] := Ltemp[(nlb, lc, nlt), (lb, lt)] (nlb ∈ 1:maximum(p_lb), lc ∈ 1:maximum(p_l), lb ∈ 1:slb)
    Ltemp = permutedims(Ltemp, (4, 1, 2, 5, 3))

    @cast Ltemp[(lc, nlt), lt, (nlb, lb)] := Ltemp[lc, nlt, lt, nlb, lb]
    Ltemp = attach_central_left(Ltemp, h, Val(:c))
    @cast Ltemp[lc, nlt, lt, nlb, lb] := Ltemp[(lc, nlt), lt, (nlb, lb)] (nlt ∈ 1:maximum(p_lb), lb ∈ 1:maximum(p_lt))
    @tensor Ltempnew[bb, ar, c, ab, br] := Ltemp[bl, at, c, al, bt] * A4[al, at, ar, ab] * B4[bl, bt, br, bb] order = (bl, bt, al, at)

    slb = size(Ltempnew, 1)
    prs = projectors_to_sparse_transposed(p_rb, p_r, p_rt, tLE) 

    Ltempnew = permutedims(Ltempnew, (5, 3, 2, 1, 4)) #[(nbp, nc, ntp), (nb, nt)]
    @cast Ltempnew[(br, c, ar), (bb, ab)] := Ltempnew[br, c, ar, bb, ab] 
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

    @cast A4[al, at, ar, ab] := A[al, (at, ar), ab] (at ∈ 1:maximum(p_lt))
    @cast B4[bl, bt, br, bb] := B[bl, (bt, br), bb] (bt ∈ 1:maximum(p_lb))

    srt = size(RE, 1)
    REn = permutedims(RE, (2, 3, 1))
    @cast REn[rc, (rb, rt)] := REn[rc, rb, rt]
    tRE = typeof(RE)
    ps = projectors_to_sparse(p_rb, p_r, p_rt, tRE)
    Rtemp = ps * REn

    @cast Rtemp[nrb, rc, nrt, rb, rt] := Rtemp[(nrb, rc, nrt), (rb, rt)] (nrb ∈ 1:maximum(p_rb), rc ∈ 1:maximum(p_r), rt ∈ 1:srt)
    Rtemp = permutedims(Rtemp, (5, 3, 2, 4, 1))

    @cast Rtemp[(rt, nrt), rc, (rb, nrb)] :=  Rtemp[rt, nrt, rc, rb, nrb]
    Rtemp = attach_central_right(Rtemp, h, Val(:n))
    @cast Rtemp[rt, nrt, rc, rb, nrb] := Rtemp[(rt, nrt), rc, (rb, nrb)] (nrb ∈ 1:maximum(p_rb), nrt ∈ 1:maximum(p_rt))
    @tensor Rtempnew[al, at, c, bl, bt] := Rtemp[ab, ar, c, bb, br] * A4[al, at, ar, ab] * B4[bl, bt, br, bb] #order = (b, tp, t, bp)

    srt = size(Rtempnew, 1)
    prs = projectors_to_sparse_transposed(p_lb, p_l, p_lt, tRE) 

    Rtempnew = permutedims(Rtempnew, (5, 3, 2, 1, 4)) #[(nbp, nc, ntp), (nb, nt)]
    @cast Rtempnew[(bl, c, at), (bt, al)] :=  Rtempnew[bl, c, at, bt, al] 
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

    @cast A4[al, at, ar, ab] := A[al, (at, ar), ab] (at ∈ 1:maximum(p_lb))
    @cast B4[bl, bt, br, bb] := B[bl, (bt, br), bb] (bt ∈ 1:maximum(p_lt))

    srt = size(RE, 1)
    REn = permutedims(RE, (2, 3, 1))
    @cast REn[rc, (rb, rt)] := REn[rc, rb, rt]
    tRE = typeof(RE)
    ps = projectors_to_sparse(p_rb, p_r, p_rt, tRE)
    Rtemp = ps * REn

    @cast Rtemp[nrb, rc, nrt, rb, rt] := Rtemp[(nrb, rc, nrt), (rb, rt)] (nrb ∈ 1:maximum(p_rb), rc ∈ 1:maximum(p_r), rt ∈ 1:srt)
    Rtemp = permutedims(Rtemp, (5, 3, 2, 4, 1))

    @cast Rtemp[(rt, nrt), rc, (rb, nrb)] :=  Rtemp[rt, nrt, rc, rb, nrb]
    Rtemp = attach_central_right(Rtemp, h, Val(:c))
    @cast Rtemp[rt, nrt, rc, rb, nrb] := Rtemp[(rt, nrt), rc, (rb, nrb)] (nrb ∈ 1:maximum(p_rb), nrt ∈ 1:maximum(p_rt))
    @tensor Rtempnew[al, bt, c, bl, at] := Rtemp[ab, br, c, bb, ar] * A4[al, at, ar, ab] * B4[bl, bt, br, bb] #order = (b, tp, t, bp)

    srt = size(Rtempnew, 1)
    prs = projectors_to_sparse_transposed(p_lb, p_l, p_lt, tRE) 

    Rtempnew = permutedims(Rtempnew, (2, 3, 5, 1, 4)) #[(nbp, nc, ntp), (nb, nt)]
    @cast Rtempnew[(bl, c, at), (bt, al)] :=  Rtempnew[bl, c, at, bt, al] 
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

    @cast B4[bl, bt, br, bb] := B[bl, (bt, br), bb] (bt ∈ 1:maximum(p_lb))

    slb = size(LE, 1)
    srt = size(RE, 1)

    LEn = permutedims(LE, (2, 1, 3))
    @cast LEn[lc, (lb, lt)] := LEn[lc, lb, lt]
    tLE = typeof(LE)
    ps = projectors_to_sparse(p_lb, p_l, p_lt, tLE)
    LL = ps * LEn

    @cast LL[nlb, lc, nlt, lb, lt] := LL[(nlb, lc, nlt), (lb, lt)] (nlb ∈ 1:maximum(p_lb), lc ∈ 1:maximum(p_l), lb ∈ 1:slb)
    LL = permutedims(LL, (4, 1, 2, 5, 3))

    @cast LL[(lc, nlt), lt, (nlb, lb)] := LL[lc, nlt, lt, nlb, lb]
    LL = attach_central_left(LL, h, Val(:n))
    @cast LL[lc, nlt, lt, nlb, lb] := LL[(lc, nlt), lt, (nlb, lb)] (nlt ∈ 1:maximum(p_lb), lb ∈ 1:maximum(p_lt))

    REn = permutedims(RE, (2, 3, 1))
    @cast REn[rc, (rb, rt)] := REn[rc, rb, rt]
    ps = projectors_to_sparse(p_rb, p_r, p_rt, tLE)
    RR = ps * REn

    @cast RR[nrb, rc, nrt, rb, rt] := RR[(nrb, rc, nrt), (rb, rt)] (nrb ∈ 1:maximum(p_rb), rc ∈ 1:maximum(p_r), rt ∈ 1:srt)
    RR = permutedims(RR, (5, 3, 2, 4, 1))

    @tensor LR[lr, tb, rt, rl] := LL[bl, bt, c, lr, tb] * RR[rl, rt, c, bb, br] * B4[bl, bt, br, bb] order = (bl, bt, br, bb, c)
    #@tensor LR[tl, tlp, trp, tr] := LL[bl, blp, cr, tl, tlp] * RR[tr, trp, cr, br, brp] * B4[bl, blp, brp, br] order = (bl, blp, brp, br, cr)

    @cast LR[l, (t, r), b] := LR[l, t, r, b]

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

    @cast B4[bl, bt, br, bb] := B[bl, (bt, br), bb] (bt ∈ 1:maximum(p_lt))

    slb = size(LE, 1)
    srt = size(RE, 1)

    LEn = permutedims(LE, (2, 1, 3))
    @cast LEn[lc, (lb, lt)] := LEn[lc, lb, lt]
    tLE = typeof(LE)
    ps = projectors_to_sparse(p_lt, p_l, p_lb, tLE)
    LL = ps * LEn

    @cast LL[nlb, lc, nlt, lb, lt] := LL[(nlb, lc, nlt), (lb, lt)] (nlb ∈ 1:maximum(p_lt), lc ∈ 1:maximum(p_l), lb ∈ 1:slb)
    LL = permutedims(LL, (4, 1, 2, 5, 3))

    @cast LL[(lc, nlt), lt, (nlb, lb)] := LL[lc, nlt, lt, nlb, lb]
    LL = attach_central_left(LL, h, Val(:c))
    @cast LL[lc, nlt, lt, nlb, lb] := LL[(lc, nlt), lt, (nlb, lb)] (nlt ∈ 1:maximum(p_lt), lb ∈ 1:maximum(p_lb))

    REn = permutedims(RE, (2, 3, 1))
    @cast REn[rc, (rb, rt)] := REn[rc, rb, rt]
    ps = projectors_to_sparse(p_rb, p_r, p_rt, tLE)
    RR = ps * REn

    @cast RR[nrb, rc, nrt, rb, rt] := RR[(nrb, rc, nrt), (rb, rt)] (nrb ∈ 1:maximum(p_rt), rc ∈ 1:maximum(p_r), rt ∈ 1:srt)
    RR = permutedims(RR, (5, 3, 2, 4, 1))
    @tensor LR[lr, tb, rt, rl] := LL[bl, bt, c, lr, tb] * RR[rl, rt, c, bb, br] * B4[bl, bt, br, bb] order = (bl, bt, br, bb, c)

    @cast LR[l, (x, y), r] := LR[l, x, y, r]

    Array(LR ./ maximum(abs.(LR)))
end
