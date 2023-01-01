export
    contract_tensor3_matrix,
    contract_matrix_tensor3,
    update_reduced_env_right

#TODO to be removed eventually
ArrayOrCuArray(L) = typeof(L) <: CuArray ? CuArray : Array

function contract_tensor3_matrix(L::Array{T, 3}, M::CentralTensor{T, 2}) where T <: Real
    e11, e12, e21, e22 = ArrayOrCuArray(L).((M.e11, M.e12, M.e21, M.e22))

    sb, _, st = size(L)
    sl1, sl2, sr1, sr2 = size(e11, 1), size(e21, 1), size(e21, 2), size(e22, 2)
    sinter = st * sb * max(sl1 * sl2 * min(sr1, sr2), sr1 * sr2 * min(sl1, sl2))

    L = reshape(permutedims(L, (1, 3, 2)), (sb * st, sl1, sl2))
    if sl1 * sl2 * sr1 * sr2 < sinter
        @cast E[l1, l2, r1, r2] := e11[l1, r1] * e21[l2, r1] * e12[l1, r2] * e22[l2, r2]
        @tensor L[tb, r1, r2] := L[tb, l1, l2] * E[l1, l2, r1, r2]
        return permutedims(reshape(L, (sb, st, sr1 * sr2)), (1, 3, 2))
    elseif sr1 <= sr2 && sl1 <= sl2
        @cast L[tb, l1, l2, r1] := L[tb, l1, l2] * e21[l2, r1]
        @tensor L[tb, l1, r1, r2] := L[tb, l1, l2, r1] * e22[l2, r2]
        @cast L[tb, l1, r1, r2] := L[tb, l1, r1, r2] * e11[l1, r1] * e12[l1, r2]
    elseif sr1 <= sr2 && sl2 <= sl1
        @cast L[tb, l1, l2, r1] := L[tb, l1, l2] * e11[l1, r1]
        @tensor L[tb, l2, r1, r2] := L[tb, l1, l2, r1] * e12[l1, r2]
        @cast L[tb, l2, r1, r2] := L[tb, l2, r1, r2] * e21[l2, r1] * e22[l2, r2]
    elseif sr2 <= sr1 && sl1 <= sl2
        @cast L[tb, l1, l2, r2] := L[tb, l1, l2] * e22[l2, r2]
        @tensor L[tb, l1, r1, r2] := L[tb, l1, l2, r2] * e21[l2, r1]
        @cast L[tb, l1, r1, r2] := L[tb, l1, r1, r2] * e11[l1, r1] * e12[l1, r2]
    else # sr2 <= sr1 && sl2 <= sl1
        @cast L[tb, l1, l2, r2] := L[tb, l1, l2] * e12[l1, r2]
        @tensor L[tb, l2, r1, r2] := L[tb, l1, l2, r2] * e11[l1, r1]
        @cast L[tb, l2, r1, r2] := L[tb, l2, r1, r2] * e21[l2, r1] * e22[l2, r2]
    end
    permutedims(reshape(sum(L, dims=2), (sb, st, sr1 * sr2)), (1, 3, 2))
end

function contract_matrix_tensor3( M::CentralTensor{T, 2}, R::Array{T, 3}) where T <: Real
    e11, e12, e21, e22 = ArrayOrCuArray(R).((M.e11, M.e12, M.e21, M.e22))

    st, _, sb = size(R)
    sl1, sl2, sr1, sr2 = size(e11, 1), size(e21, 1), size(e21, 2), size(e22, 2)
    sinter = st * sb * max(sl1 * sl2 * min(sr1, sr2), sr1 * sr2 * min(sl1, sl2))

    R = reshape(permutedims(R, (2, 1, 3)), (sr1, sr2, st * sb))
    if sl1 * sl2 * sr1 * sr2 <= sinter
        @cast E[l1, l2, r1, r2] := e11[l1, r1] * e21[l2, r1] * e12[l1, r2] * e22[l2, r2]
        @tensor R[l1, l2, tb] := E[l1, l2, r1, r2] * R[r1, r2, tb]
        return permutedims(reshape(R, (sl1 * sl2, st, sb)), (2, 1, 3))
    elseif sl1 <= sl2 && sr1 <= sr2
        @cast R[l1, r1, r2, tb] := e12[l1, r2] * R[r1, r2, tb]
        @tensor R[l1, l2, r1, tb] := e22[l2, r2] * R[l1, r1, r2, tb]
        @cast R[l1, l2, r1, tb] := e11[l1, r1] * e21[l2, r1] * R[l1, l2, r1, tb]
    elseif sl1 <= sl2 && sr2 <= sr1
        @cast R[l1, r1, r2, tb] := e11[l1, r1] * R[r1, r2, tb]
        @tensor R[l1, l2, r2, tb] := e21[l2, r1] * R[l1, r1, r2, tb]
        @cast R[l1, l2, r2, tb] := e12[l1, r2] * e22[l2, r2] * R[l1, l2, r2, tb]
    elseif sl2 <= sl1 && sr1 <= sr2
        @cast R[l2, r1, r2, tb] := e22[l2, r2] * R[r1, r2, tb]
        @tensor R[l1, l2, r1, tb] := e12[l1, r2] * R[l2, r1, r2, tb]
        @cast R[l1, l2, r1, tb] := e11[l1, r1] * e21[l2, r1] * R[l1, l2, r1, tb]
    else # sl2 <= sl1 && sr2 <= sr1
        @cast R[l2, r1, r2, tb] := e21[l2, r1] * R[r1, r2, tb]
        @tensor R[l1, l2, r2, tb] := e11[l1, r1] * R[l2, r1, r2, tb]
        @cast R[l1, l2, r2, tb] := e12[l1, r2] * e22[l2, r2] * R[l1, l2, r2, tb]
    end
    permutedims(reshape(sum(R, dims=3), (sl1 * sl2, st, sb)), (2, 1, 3))
end

function update_reduced_env_right(RR::Array{T, 2}, M::CentralTensor{T, 2}) where T <: Real
    dropdims(contract_matrix_tensor3(M, reshape(RR, size(RR, 1), size(RR, 2), 1)), dims=3)
end
