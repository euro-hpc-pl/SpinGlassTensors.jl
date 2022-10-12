function update_env_left(
    LE::S, A::S, M::T, B::S, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}
    h = M.con
    if typeof(h) == SparseCentralTensor
        h = cuda_dense_central_tensor(h)
    else
        h = CUDA.CuArray(h)
    end
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs
    @cast A4[x, k, l, y] := A[x, (k, l), y] (k ∈ 1:maximum(p_lt))
    @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lb))
    
    A4 = CUDA.CuArray(A4)
    B4 = CUDA.CuArray(B4)
    
    ps = projectors_to_cusparse(p_lb, p_l, p_lt)

    (a,b,c) = size(LE)
    LE = permutedims(CUDA.CuArray(LE), (2, 1, 3))
    @cast LEn[x, (y, z)] := LE[x, y, z]
    Ltemp = ps * LEn 

    @cast Ltemp[nbp, nc, ntp, nb, nt] := Ltemp[(nbp, nc, ntp), (nb, nt)] (nbp ∈ 1:maximum(p_lb), nc ∈ 1:maximum(p_l), nb ∈ 1:a)
    Ltemp = permutedims(CUDA.CuArray(Ltemp), (4, 1, 2, 5, 3))

    @tensor Ltempnew[nb, nbp, nc, nt, ntp] := Ltemp[b, bp, oc, t, tp] * A4[t, tp, ntp, nt] * B4[b, bp, nbp, nb] * h[oc, nc]
    
    a = size(Ltempnew, 1)
    prs = projectors_to_cusparse_transposed(p_rb, p_r, p_rt) 

    Ltempnew = permutedims(CUDA.CuArray(Ltempnew), (2, 3, 5, 1, 4)) #[(nbp, nc, ntp), (nb, nt)]
    @cast Ltempnew[(nbp, nc, ntp), (nb, nt)] :=  Ltempnew[nbp,  nc, ntp, nb, nt] 
    Lnew = prs * Ltempnew  #[cc, (nb, nt)]  
    Lnew = permutedims(CUDA.CuArray(Lnew), (2, 1))  #[(nb, nt), cc]
    @cast Lnew[nb, nt, cc] := Lnew[(nb, nt), cc] (nb ∈ 1:a)

    Array(permutedims(Lnew, (1, 3, 2)) ./ maximum(abs.(Lnew)))

end

function update_env_left(
    LE::S, A::S, M::T, B::S, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}
    h = M.con
    if typeof(h) == SparseCentralTensor
        h = cuda_dense_central_tensor(h)
    else
        h = CUDA.CuArray(h)
    end
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

    @tensor Ltempnew[nb, ntp, nc, nt, nbp] := Ltemp[b, bp, oc, t, tp] * A4[t, bp, ntp, nt] * B4[b, tp, nbp, nb] * h[oc, nc]
    
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
    if typeof(h) == SparseCentralTensor
        h = cuda_dense_central_tensor(h)
    else
        h = CUDA.CuArray(h)
    end
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
    @tensor Rtempnew[nt, ntp, nc, nb, nbp] := Rtemp[t, tp, oc, b, bp] * A4[nt, ntp, tp, t] * B4[nb, nbp, bp, b] * h[nc, oc]

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
    if typeof(h) == SparseCentralTensor
        h = cuda_dense_central_tensor(h)
    else
        h = CUDA.CuArray(h)
    end
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
    @tensor Rtempnew[nt, nbp, nc, nb, ntp] := Rtemp[t, tp, oc, b, bp] * A4[nt, ntp, bp, t] * B4[nb, nbp, tp, b] * h[nc, oc]

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
    if typeof(h) == SparseCentralTensor
        h = cuda_dense_central_tensor(h)
    else
        h = CUDA.CuArray(h)
    end
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    B = CUDA.CuArray(B)
    @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lb))
    
    pls = projectors_to_cusparse(p_lb, p_l, p_lt)
    (a,b,c) = size(LE)
    LE = permutedims(CUDA.CuArray(LE), (2, 1, 3))
    @cast LEn[x, (y, z)] := LE[x, y, z]
    LL = pls * LEn 

    @cast LL[nbp, nc, ntp, nb, nt] := LL[(nbp, nc, ntp), (nb, nt)] (nbp ∈ 1:maximum(p_lb), nc ∈ 1:maximum(p_l), nb ∈ 1:a)
    LL = permutedims(CUDA.CuArray(LL), (4, 1, 2, 5, 3))

    prs = projectors_to_cusparse(p_rb, p_r, p_rt)
    (a,b,c) = size(RE)
    RE = permutedims(CUDA.CuArray(RE), (2, 3, 1))
    @cast REn[x, (y, z)] := RE[x, y, z]
    RR = prs * REn 
    @cast RR[nbp, nc, ntp, nb, nt] := RR[(nbp, nc, ntp), (nb, nt)] (nbp ∈ 1:maximum(p_rb), nc ∈ 1:maximum(p_r), nt ∈ 1:a)
    RR = permutedims(CUDA.CuArray(RR), (5, 3, 2, 4, 1))

    @tensor LR[tl, tlp, trp, tr] := LL[bl, blp, cl, tl, tlp] * RR[tr, trp, cr, br, brp] * B4[bl, blp, brp, br] * h[cl, cr] order = (cl, bl, blp, brp, br, cr)
    @cast LR[l, (x, y), r] := LR[l, x, y, r]

    Array(LR ./ maximum(abs.(LR)))
end

function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}
    h = M.con
    if typeof(h) == SparseCentralTensor
        h = cuda_dense_central_tensor(h)
    else
        h = CUDA.CuArray(h)
    end
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    B = CUDA.CuArray(B)
    @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lt))

    pls = projectors_to_cusparse(p_lt, p_l, p_lb)
    (a,b,c) = size(LE)
    LE = permutedims(CUDA.CuArray(LE), (2, 1, 3))
    @cast LEn[x, (y, z)] := LE[x, y, z]
    LL = pls * LEn 
    @cast LL[nbp, nc, ntp, nb, nt] := LL[(nbp, nc, ntp), (nb, nt)] (nbp ∈ 1:maximum(p_lt), nc ∈ 1:maximum(p_l), nb ∈ 1:a)
    LL = permutedims(CUDA.CuArray(LL), (4, 1, 2, 5, 3))

    prs = projectors_to_cusparse(p_rt, p_r, p_rb)
    (a,b,c) = size(RE)
    RE = permutedims(CUDA.CuArray(RE), (2, 3, 1))
    @cast REn[x, (y, z)] := RE[x, y, z]
    RR = prs * REn 
    @cast RR[nbp, nc, ntp, nb, nt] := RR[(nbp, nc, ntp), (nb, nt)] (nbp ∈ 1:maximum(p_rt), nc ∈ 1:maximum(p_r), nt ∈ 1:a)
    RR = permutedims(CUDA.CuArray(RR), (5, 3, 2, 4, 1))

    @tensor LR[tl, tlp, trp, tr] := LL[bl, blp, cl, tl, tlp] * RR[tr, trp, cr, br, brp] * B4[bl, blp, brp, br] * h[cl, cr] order = (cl, bl, blp, brp, br, cr)
    @cast LR[l, (x, y), r] := LR[l, x, y, r]

    Array(LR ./ maximum(abs.(LR)))
end

