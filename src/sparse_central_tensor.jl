export 
    attach_central_left,
    attach_central_right

"""
$(TYPEDSIGNATURES)
"""
function attach_central_left(
    L::S, M::T
) where {S <: AbstractArray{Float64, 3}, T <: SparseCentralTensor}
    if typeof(L) <: CuArray
        e11, e12, e21, e22 = CUDA.CuArray.((M.e11, M.e12, M.e21, M.e22))
    else
        e11, e12, e21, e22 = M.e11, M.e12, M.e21, M.e22
    end
    sb, _, st = size(L)
    sl1, sl2, sr1, sr2 = size(e11, 1), size(e21, 1), size(e21, 2), size(e22, 2)
    L = permutedims(L, (1, 3, 2))
    L = reshape(L, (sb * st, sl1, sl2))
    sinter = st * sb * max(sl1 * sl2 * min(sr1, sr2), sr1 * sr2 * min(sl1, sl2))
    if sl1 * sl2 * sr1 * sr2 < sinter
        @cast E[l1, l2, r1, r2] := e11[l1, r1] * e21[l2, r1] * e12[l1, r2] * e22[l2, r2]
        @tensor L[tb, r1, r2] := L[tb, l1, l2] * E[l1, l2, r1, r2]
    elseif sr1 <= sr2 && sl1 <= sl2
        @cast L[tb, l1, l2, r1] := L[tb, l1, l2] * e21[l2, r1]
        @tensor L[tb, l1, r1, r2] := L[tb, l1, l2, r1] * e22[l2, r2]
        @cast L[tb, l1, r1, r2] := L[tb, l1, r1, r2] * e11[l1, r1] * e12[l1, r2]
        L = dropdims(sum(L, dims=2), dims=2)
    elseif sr1 <= sr2 && sl2 <= sl1
        @cast L[tb, l1, l2, r1] := L[tb, l1, l2] * e11[l1, r1]
        @tensor L[tb, l2, r1, r2] := L[tb, l1, l2, r1] * e12[l1, r2]
        @cast L[tb, l2, r1, r2] := L[tb, l2, r1, r2] * e21[l2, r1] * e22[l2, r2]
        L = dropdims(sum(L, dims=2), dims=2)
    elseif sr2 <= sr1 && sl1 <= sl2
        @cast L[tb, l1, l2, r2] := L[tb, l1, l2] * e22[l2, r2]
        @tensor L[tb, l1, r1, r2] := L[tb, l1, l2, r2] * e21[l2, r1]
        @cast L[tb, l1, r1, r2] := L[tb, l1, r1, r2] * e11[l1, r1] * e12[l1, r2]
        L = dropdims(sum(L, dims=2), dims=2)
    else # sr2 <= sr1 && sl2 <= sl1
        @cast L[tb, l1, l2, r2] := L[tb, l1, l2] * e12[l1, r2]
        @tensor L[tb, l2, r1, r2] := L[tb, l1, l2, r2] * e11[l1, r1]
        @cast L[tb, l2, r1, r2] := L[tb, l2, r1, r2] * e21[l2, r1] * e22[l2, r2]
        L = dropdims(sum(L, dims=2), dims=2)
    end
    L = reshape(L, (sb, st, sr1 * sr2))
    L = permutedims(L, (1, 3, 2))
    L
end

"""
$(TYPEDSIGNATURES)
"""
function attach_central_right(
    R::S, M::T
) where {S <: AbstractArray{Float64, 3}, T <: SparseCentralTensor}
    if typeof(R) <: CuArray
        e11, e12, e21, e22 = CUDA.CuArray.((M.e11, M.e12, M.e21, M.e22))
    else
        e11, e12, e21, e22 = M.e11, M.e12, M.e21, M.e22
    end
    st, _, sb = size(R)
    sl1, sl2, sr1, sr2 = size(e11, 1), size(e21, 1), size(e21, 2), size(e22, 2)
    R = permutedims(R, (2, 1, 3))
    R = reshape(R, (sr1, sr2, st * sb))

    sinter = st * sb * max(sl1 * sl2 * min(sr1, sr2), sr1 * sr2 * min(sl1, sl2))
    if sl1 * sl2 * sr1 * sr2 < sinter
        @cast E[l1, l2, r1, r2] := e11[l1, r1] * e21[l2, r1] * e12[l1, r2] * e22[l2, r2]
        @tensor R[l1, l2, tb] := E[l1, l2, r1, r2] * R[r1, r2, tb]
    elseif sl1 <= sl2 && sr1 <= sr2
        @cast R[l1, r1, r2, tb] := e12[l1, r2] * R[r1, r2, tb]
        @tensor R[l1, l2, r1, tb] := e22[l2, r2] * R[l1, r1, r2, tb]
        @cast R[l1, l2, r1, tb] := e11[l1, r1] * e21[l2, r1] * R[l1, l2, r1, tb]
        R = dropdims(sum(R, dims=3), dims=3)
    elseif sl1 <= sl2 && sr2 <= sr1
        @cast R[l1, r1, r2, tb] := e11[l1, r1] * R[r1, r2, tb]
        @tensor R[l1, l2, r2, tb] := e21[l2, r1] * R[l1, r1, r2, tb]
        @cast R[l1, l2, r2, tb] := e12[l1, r2] * e22[l2, r2] * R[l1, l2, r2, tb]
        R = dropdims(sum(R, dims=3), dims=3)
    elseif sl2 <= sl1 && sr1 <= sr2
        @cast R[l2, r1, r2, tb] := e22[l2, r2] * R[r1, r2, tb]
        @tensor R[l1, l2, r1, tb] := e12[l1, r2] * R[l2, r1, r2, tb]
        @cast R[l1, l2, r1, tb] := e11[l1, r1] * e21[l2, r1] * R[l1, l2, r1, tb]
        R = dropdims(sum(R, dims=3), dims=3)
    else # sl2 <= sl1 && sr2 <= sr1
        @cast R[l2, r1, r2, tb] := e21[l2, r1] * R[r1, r2, tb]
        @tensor R[l1, l2, r2, tb] := e11[l1, r1] * R[l2, r1, r2, tb]
        @cast R[l1, l2, r2, tb] := e12[l1, r2] * e22[l2, r2] * R[l1, l2, r2, tb]
        R = dropdims(sum(R, dims=3), dims=3)
    end
    R = reshape(R, (sl1 * sl2, st, sb))
    R = permutedims(R, (2, 1, 3))
    R
end

