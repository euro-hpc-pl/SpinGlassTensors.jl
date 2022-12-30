export
    contract_tensor3_matrix,
    contract_matrix_tensor3,
    update_reduced_env_right

function contract_tensor3_matrix(L::CuArray{T, 3}, M::CentralTensor{T, 2}) where T <: Real
    sb, _, st = size(L)
    sl1, sl2, sr1, sr2 = size(M.e11, 1), size(M.e21, 1), size(M.e21, 2), size(M.e22, 2)
    sinter = st * sb * max(sl1 * sl2 * min(sr1, sr2), sr1 * sr2 * min(sl1, sl2))
    L = reshape(permutedims(L, (1, 3, 2)), (sb * st, sl1, sl2))
    if sl1 * sl2 * sr1 * sr2 < sinter
        @cast E[l1, l2, r1, r2] := M.e11[l1, r1] * M.e21[l2, r1] * M.e12[l1, r2] * M.e22[l2, r2]
        @tensor L[tb, r1, r2] := L[tb, l1, l2] * E[l1, l2, r1, r2]
        return permutedims(reshape(L, (sb, st, sr1 * sr2)), (1, 3, 2))
    elseif sr1 <= sr2 && sl1 <= sl2
        @cast L[tb, l1, l2, r1] := L[tb, l1, l2] * M.e21[l2, r1]
        @tensor L[tb, l1, r1, r2] := L[tb, l1, l2, r1] * M.e22[l2, r2]
        L .*= reshape(M.e11, 1, sl1, sr1, 1)  # [tb, l1, r1, r2] .* [:, l1, r1, :]
        L .*= reshape(M.e12, 1, sl1, 1, sr2)  # [tb, l1, r1, r2] .* [:, l1, :, r2]
    elseif sr1 <= sr2 && sl2 <= sl1
        @cast L[tb, l1, l2, r1] := L[tb, l1, l2] * M.e11[l1, r1]
        @tensor L[tb, l2, r1, r2] := L[tb, l1, l2, r1] * M.e12[l1, r2]
        L .*= reshape(M.e21, 1, sl2, sr1, 1)  # [tb, l2, r1, r2] .* [:, l2, r1, :]
        L .*= reshape(M.e22, 1, sl2, 1, sr2)  # [tb, l2, r1, r2] .* [:, l2, :, r2]
    elseif sr2 <= sr1 && sl1 <= sl2
        @cast L[tb, l1, l2, r2] := L[tb, l1, l2] * M.e22[l2, r2]
        @tensor L[tb, l1, r1, r2] := L[tb, l1, l2, r2] * M.e21[l2, r1]
        L .*= reshape(M.e11, 1, sl1, sr1, 1)  # [tb, l1, r1, r2] .* [:, l1, r1, :]
        L .*= reshape(M.e12, 1, sl1, 1, sr2)  # [tb, l1, r1, r2] .* [:, l1, :, r2]
    else # sr2 <= sr1 && sl2 <= sl1
        @cast L[tb, l1, l2, r2] := L[tb, l1, l2] * M.e12[l1, r2]
        @tensor L[tb, l2, r1, r2] := L[tb, l1, l2, r2] * M.e11[l1, r1]
        L .*= reshape(M.e21, 1, sl2, sr1, 1)  # [tb, l2, r1, r2] .* [:, l2, r1, :]
        L .*= reshape(M.e22, 1, sl2, 1, sr2)  # [tb, l2, r1, r2] .* [:, l2, :, r2]
    end
    permutedims(reshape(sum(L, dims=2), (sb, st, sr1 * sr2)), (1, 3, 2))
end

function contract_matrix_tensor3( M::CentralTensor{T, 2}, R::CuArray{T, 3}) where T <: Real
    st, _, sb = size(R)
    sl1, sl2, sr1, sr2 = size(M.e11, 1), size(M.e21, 1), size(M.e21, 2), size(M.e22, 2)
    sinter = st * sb * max(sl1 * sl2 * min(sr1, sr2), sr1 * sr2 * min(sl1, sl2))
    R = reshape(permutedims(R, (2, 1, 3)), (sr1, sr2, st * sb))
    if sl1 * sl2 * sr1 * sr2 <= sinter
        @cast E[l1, l2, r1, r2] := M.e11[l1, r1] * M.e21[l2, r1] * M.e12[l1, r2] * M.e22[l2, r2]
        @tensor R[l1, l2, tb] := E[l1, l2, r1, r2] * R[r1, r2, tb]
        return permutedims(reshape(R, (sl1 * sl2, st, sb)), (2, 1, 3))
    elseif sl1 <= sl2 && sr1 <= sr2
        @cast R[l1, r1, r2, tb] := M.e12[l1, r2] * R[r1, r2, tb]
        @tensor R[l1, l2, r1, tb] := M.e22[l2, r2] * R[l1, r1, r2, tb]
        R .*= reshape(M.e11, sl1, 1, sr1, 1)  # [l1, l2, r1, tb] .* [l1, :, r1, :]
        R .*= reshape(M.e21, 1, sl2, sr1, 1)  # [l1, l2, r1, tb] .* [:, l2, r1, :]
    elseif sl1 <= sl2 && sr2 <= sr1
        @cast R[l1, r1, r2, tb] := M.e11[l1, r1] * R[r1, r2, tb]
        @tensor R[l1, l2, r2, tb] := M.e21[l2, r1] * R[l1, r1, r2, tb]
        R .*= reshape(M.e12, sl1, 1, sr2, 1)  # [l1, l2, r2, tb] .* [l1, :, r2, :]
        R .*= reshape(M.e22, 1, sl2, sr2, 1)  # [l1, l2, r2, tb] .* [:, l2, r2, :]
    elseif sl2 <= sl1 && sr1 <= sr2
        @cast R[l2, r1, r2, tb] := M.e22[l2, r2] * R[r1, r2, tb]
        @tensor R[l1, l2, r1, tb] := M.e12[l1, r2] * R[l2, r1, r2, tb]
        R .*= reshape(M.e11, sl1, 1, sr1, 1)  # [l1, l2, r1, tb] .* [l1, :, r1, :]
        R .*= reshape(M.e21, 1, sl2, sr1, 1)  # [l1, l2, r1, tb] .* [:, l2, r1, :]
    else # sl2 <= sl1 && sr2 <= sr1
        @cast R[l2, r1, r2, tb] := M.e21[l2, r1] * R[r1, r2, tb]
        @tensor R[l1, l2, r2, tb] := M.e11[l1, r1] * R[l2, r1, r2, tb]
        R .*= reshape(M.e12, sl1, 1, sr2, 1)  # [l1, l2, r2, tb] .* [l1, :, r2, :]
        R .*= reshape(M.e22, 1, sl2, sr2, 1)  # [l1, l2, r2, tb] .* [:, l2, r2, :]
    end
    permutedims(reshape(sum(R, dims=3), (sl1 * sl2, st, sb)), (2, 1, 3))
end

function update_reduced_env_right(RR::CuArray{T, 2}, M::CentralTensor{T, 2}) where T <: Real
    RR = reshape(RR, size(RR, 1), size(RR, 2), 1)
    RR = contract_matrix_tensor3(M, RR)
    dropdims(RR, dims=3)
end
