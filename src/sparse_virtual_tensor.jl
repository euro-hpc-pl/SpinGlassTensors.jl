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

    @cast A4[x, k, l, y] := A[x, (k, l), y] (k ∈ 1:maximum(p_lt))
    @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lb))

    sb = size(LE, 1)
    LEn = permutedims(LE, (2, 1, 3))
    @cast LEn[lc, (lb, lt)] := LEn[lc, lb, lt]
    ps = projectors_to_cusparse(p_lb, p_l, p_lt)
    Ltemp = ps * LEn

    @cast Ltemp[nbp, nc, ntp, nb, nt] := Ltemp[(nbp, nc, ntp), (nb, nt)] (nbp ∈ 1:maximum(p_lb), nc ∈ 1:maximum(p_l), nb ∈ 1:sb)
    Ltemp = permutedims(Ltemp, (4, 1, 2, 5, 3))

    @cast Ltemp[(b, bp), oc, (t, tp)] := Ltemp[b, bp, oc, t, tp]
    Ltemp = attach_central_left(Ltemp, h, Val(:n))
    @cast Ltemp[nb, nbp, nc, nt, ntp] := Ltemp[(nb, nbp), nc, (nt, ntp)] (nbp ∈ 1:maximum(p_lb), ntp ∈ 1:maximum(p_lt))
    @tensor Ltempnew[nb, nbp, nc, nt, ntp] := Ltemp[b, bp, nc, t, tp] * A4[t, tp, ntp, nt] * B4[b, bp, nbp, nb] order = (b, bp, t, tp)

    sb = size(Ltempnew, 1)
    prs = projectors_to_cusparse_transposed(p_rb, p_r, p_rt) 

    Ltempnew = permutedims(Ltempnew, (2, 3, 5, 1, 4)) #[(nbp, nc, ntp), (nb, nt)]
    @cast Ltempnew[(nbp, nc, ntp), (nb, nt)] := Ltempnew[nbp,  nc, ntp, nb, nt] 
    Lnew = prs * Ltempnew  #[cc, (nb, nt)]

    @cast Lnew[cc, nb, nt] := Lnew[cc, (nb, nt)] (nb ∈ 1:sb)
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

    @cast A4[x, k, l, y] := A[x, (k, l), y] (k ∈ 1:maximum(p_lb))
    @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lt))
    A4 = CUDA.CuArray(A4)
    B4 = CUDA.CuArray(B4)

    ps = projectors_to_cusparse(p_lb, p_l, p_lt)
    (a,b,c) = size(LE)
    LE = permutedims(CUDA.CuArray(LE), (2, 1, 3))
    @cast LEn[x, (y, z)] := LE[x, y, z]
    Ltemp = ps * LEn 

    @cast Ltemp[nbp, nc, ntp, nb, nt] := Ltemp[(nbp, nc, ntp), (nb, nt)] (nbp ∈ 1:maximum(p_lb), nc ∈ 1:maximum(p_l), nb ∈ 1:a)
    Ltemp = permutedims(CUDA.CuArray(Ltemp), (4, 1, 2, 5, 3))
    @cast Ltemp[(b, bp), oc, (t, tp)] :=  Ltemp[b, bp, oc, t, tp]
    Ltemp = attach_central_left(Array(Ltemp), h, Val(:n))
    Ltemp = CUDA.CuArray(Ltemp)
    @cast Ltemp[nb, nbp, nc, nt, ntp] := Ltemp[(nb, nbp), nc, (nt, ntp)] (nbp ∈ 1:maximum(p_lb), ntp ∈ 1:maximum(p_lt))
    @tensor Ltempnew[nb, ntp, nc, nt, nbp] := Ltemp[b, bp, nc, t, tp] * A4[t, bp, ntp, nt] * B4[b, tp, nbp, nb]
    
    a = size(Ltempnew, 1)
    prs = projectors_to_cusparse_transposed(p_rb, p_r, p_rt) 
    Ltempnew = permutedims(CUDA.CuArray(Ltempnew), (5, 3, 2, 1, 4)) #[(nbp, nc, ntp), (nb, nt)]
    @cast Ltempnew[(nbp, nc, ntp), (nb, nt)] :=  Ltempnew[nbp,  nc, ntp, nb, nt] 
    Lnew = prs * Ltempnew  #[cc, (nb, nt)]  
    Lnew = permutedims(CUDA.CuArray(Lnew), (2, 1))  #[(nb, nt), cc]
    @cast Lnew[nb, nt, cc] := Lnew[(nb, nt), cc] (nb ∈ 1:a)
    Array(permutedims(Lnew, (1, 3, 2)) ./ maximum(abs.(Lnew)))
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_right(
    RE::S, A::S, M::T, B::S, ::Val{:n}
) where {T <: SparseVirtualTensor, S <: AbstractArray{Float64,3}}
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    @cast A4[x, k, l, y] := A[x, (k, l), y] (k ∈ 1:maximum(p_lt))
    @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lb))
    A4 = CUDA.CuArray(A4)
    B4 = CUDA.CuArray(B4)

    ps = projectors_to_cusparse(p_rb, p_r, p_rt)
    (a,b,c) = size(RE)
    RE = permutedims(CUDA.CuArray(RE), (2, 3, 1))
    @cast REn[x, (y, z)] := RE[x, y, z]
    Rtemp = ps * REn #nc, b, t

    @cast Rtemp[nbp, nc, ntp, nb, nt] := Rtemp[(nbp, nc, ntp), (nb, nt)] (nbp ∈ 1:maximum(p_rb), nc ∈ 1:maximum(p_r), nt ∈ 1:a)
    Rtemp = permutedims(CUDA.CuArray(Rtemp), (5, 3, 2, 4, 1))

    @cast Rtemp[(t, tp), oc, (b, bp)] :=  Rtemp[t, tp, oc, b, bp]
    Rtemp = attach_central_right(Array(Rtemp), h, Val(:n))
    Rtemp = CUDA.CuArray(Rtemp)
    @cast Rtemp[nt, ntp, nc, nb, nbp] := Rtemp[(nt, ntp), nc, (nb, nbp)] (nbp ∈ 1:maximum(p_rb), ntp ∈ 1:maximum(p_rt))
   
    @tensor Rtempnew[nt, ntp, nc, nb, nbp] := Rtemp[t, tp, nc, b, bp] * A4[nt, ntp, tp, t] * B4[nb, nbp, bp, b]

    a = size(Rtempnew, 1)

    prs = projectors_to_cusparse_transposed(p_lb, p_l, p_lt) 
    Rtempnew = permutedims(CUDA.CuArray(Rtempnew), (5, 3, 2, 1, 4)) #[(nbp, nc, ntp), (nb, nt)]
    @cast Rtempnew[(ntp, nc, nbp), (nt, nb)] :=  Rtempnew[ntp,  nc, nbp, nt, nb] 
    Rnew = prs * Rtempnew  #[cc, (nb, nt)]  
    Rnew = permutedims(CUDA.CuArray(Rnew), (2, 1))  #[(nb, nt), cc]
    @cast Rnew[nt, nb, cc] := Rnew[(nb, nt), cc] (nt ∈ 1:a)
    
    Array(permutedims(Rnew, (1, 3, 2)) ./ maximum(abs.(Rnew)))
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_right(
    RE::S, A::S, M::T, B::S, ::Val{:c}
) where {T <: SparseVirtualTensor, S <: AbstractArray{Float64,3}}
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    @cast A4[x, k, l, y] := A[x, (k, l), y] (k ∈ 1:maximum(p_lb))
    @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lt))
    A4 = CUDA.CuArray(A4)
    B4 = CUDA.CuArray(B4)

    ps = projectors_to_cusparse(p_rb, p_r, p_rt)
    (a,b,c) = size(RE)
    RE = permutedims(CUDA.CuArray(RE), (2, 3, 1))
    @cast REn[x, (y, z)] := RE[x, y, z]
    Rtemp = ps * REn #nc, b, t

    @cast Rtemp[nbp, nc, ntp, nb, nt] := Rtemp[(nbp, nc, ntp), (nb, nt)] (nbp ∈ 1:maximum(p_rb), nc ∈ 1:maximum(p_r), nt ∈ 1:a)
    Rtemp = permutedims(Rtemp, (5, 3, 2, 4, 1))

    @cast Rtemp[(t, tp), oc, (b, bp)] :=  Rtemp[t, tp, oc, b, bp]
    Rtemp = attach_central_right(Array(Rtemp), h, Val(:n))
    Rtemp = CUDA.CuArray(Rtemp)
    @cast Rtemp[nt, ntp, nc, nb, nbp] := Rtemp[(nt, ntp), nc, (nb, nbp)] (nbp ∈ 1:maximum(p_rb), ntp ∈ 1:maximum(p_rt))
    @tensor Rtempnew[nt, nbp, nc, nb, ntp] := Rtemp[t, tp, nc, b, bp] * A4[nt, ntp, bp, t] * B4[nb, nbp, tp, b]

    a = size(Rtempnew, 1)

    prs = projectors_to_cusparse_transposed(p_lb, p_l, p_lt) 
    Rtempnew = permutedims(CUDA.CuArray(Rtempnew), (2, 3, 5, 1, 4)) #[(nbp, nc, ntp), (nb, nt)]
    @cast Rtempnew[(ntp, nc, nbp), (nt, nb)] :=  Rtempnew[ntp,  nc, nbp, nt, nb] 
    Rnew = prs * Rtempnew  #[cc, (nb, nt)]  
    Rnew = permutedims(CUDA.CuArray(Rnew), (2, 1))  #[(nb, nt), cc]
    @cast Rnew[nt, nb, cc] := Rnew[(nb, nt), cc] (nt ∈ 1:a)
    
    Array(permutedims(Rnew, (1, 3, 2)) ./ maximum(abs.(Rnew)))
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
    @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lb))
    
    pls = projectors_to_cusparse(p_lb, p_l, p_lt)
    (al, bl, cl) = size(LE)
    LE = permutedims(CUDA.CuArray(LE), (2, 1, 3))
    @cast LEn[x, (y, z)] := LE[x, y, z]
    LL = pls * LEn 

    @cast LL[nbp, nc, ntp, nb, nt] := LL[(nbp, nc, ntp), (nb, nt)] (nbp ∈ 1:maximum(p_lb), nc ∈ 1:maximum(p_l), nb ∈ 1:al)
    LL = permutedims(CUDA.CuArray(LL), (4, 1, 2, 5, 3))

    prs = projectors_to_cusparse(p_rb, p_r, p_rt)
    (ar, br, cr) = size(RE)
    RE = permutedims(CUDA.CuArray(RE), (2, 3, 1))
    @cast REn[x, (y, z)] := RE[x, y, z]
    RR = prs * REn 
    @cast RR[nbp, nc, ntp, nb, nt] := RR[(nbp, nc, ntp), (nb, nt)] (nbp ∈ 1:maximum(p_rb), nc ∈ 1:maximum(p_r), nt ∈ 1:ar)
    RR = permutedims(CUDA.CuArray(RR), (5, 3, 2, 4, 1))

    @cast LL[(bl, blp), cl, (tl, tlp)] :=  LL[bl, blp, cl, tl, tlp]
    LL = attach_central_left(Array(LL), h, Val(:n))
    LL = CUDA.CuArray(LL)
    @cast LL[bl, blp, cr, tl, tlp] := LL[(bl, blp), cr, (tl, tlp)] (blp ∈ 1:maximum(p_lb), tlp ∈ 1:maximum(p_lt))
    @tensor LR[tl, tlp, trp, tr] := LL[bl, blp, cr, tl, tlp] * RR[tr, trp, cr, br, brp] * B4[bl, blp, brp, br] order = (bl, blp, brp, br, cr)
    @cast LR[l, (x, y), r] := LR[l, x, y, r]

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
    @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lt))

    pls = projectors_to_cusparse(p_lt, p_l, p_lb)
    (al, bl, cl) = size(LE)
    LE = permutedims(CUDA.CuArray(LE), (2, 1, 3))
    @cast LEn[x, (y, z)] := LE[x, y, z]
    LL = pls * LEn 
    @cast LL[nbp, nc, ntp, nb, nt] := LL[(nbp, nc, ntp), (nb, nt)] (nbp ∈ 1:maximum(p_lt), nc ∈ 1:maximum(p_l), nb ∈ 1:al)
    LL = permutedims(CUDA.CuArray(LL), (4, 1, 2, 5, 3))

    prs = projectors_to_cusparse(p_rt, p_r, p_rb)
    (ar, br, cr) = size(RE)
    RE = permutedims(CUDA.CuArray(RE), (2, 3, 1))
    @cast REn[x, (y, z)] := RE[x, y, z]
    RR = prs * REn 
    @cast RR[nbp, nc, ntp, nb, nt] := RR[(nbp, nc, ntp), (nb, nt)] (nbp ∈ 1:maximum(p_rt), nc ∈ 1:maximum(p_r), nt ∈ 1:ar)
    RR = permutedims(CUDA.CuArray(RR), (5, 3, 2, 4, 1))
    @cast LL[(bl, blp), cl, (tl, tlp)] :=  LL[bl, blp, cl, tl, tlp]
    LL = attach_central_left(Array(LL), h, Val(:c))
    LL = CUDA.CuArray(LL)
    @cast LL[bl, blp, cr, tl, tlp] := LL[(bl, blp), cr, (tl, tlp)] (blp ∈ 1:maximum(p_lt), tlp ∈ 1:maximum(p_lb))
    @tensor LR[tl, tlp, trp, tr] := LL[bl, blp, cr, tl, tlp] * RR[tr, trp, cr, br, brp] * B4[bl, blp, brp, br] order = (bl, blp, brp, br, cr)
    @cast LR[l, (x, y), r] := LR[l, x, y, r] 

    Array(LR ./ maximum(abs.(LR)))
end
