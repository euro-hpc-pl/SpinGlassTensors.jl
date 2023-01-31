# contractions with CentralTensor on CPU and CUDA

export
    contract_tensor3_matrix,
    contract_matrix_tensor3,
    update_reduced_env_right

function contract_tensor3_matrix(LE::Union{Array{R, 3}, CuArray{R, 3}}, M::CentralTensor{R, 2}) where R <: Real
    contract_tensor3_central(LE, M.e11, M.e12, M.e21, M.e22)
end

function contract_matrix_tensor3(M::CentralTensor{R, 2}, RE::Union{Array{R, 3}, CuArray{R, 3}}) where R <: Real
    contract_tensor3_central(RE, M.e11', M.e21', M.e12', M.e22')
end

function update_reduced_env_right(RR::Union{Array{R, 2}, CuArray{R, 2}}, M::CentralTensor{R, 2}) where R <: Real
    RR = reshape(RR, size(RR, 1), 1, size(RR, 2))
    dropdims(contract_matrix_tensor3(M, RR), dims=2)
end

function contract_tensor3_central(LE, e11, e12, e21, e22)
    sb, st = size(LE)
    sbt = sb * st
    sl1, sl2, sr1, sr2 = size(e11, 1), size(e22, 1), size(e11, 2), size(e22, 2)
    sinter = sbt * max(sl1 * sl2 * min(sr1, sr2), sr1 * sr2 * min(sl1, sl2))
    if sl1 * sl2 * sr1 * sr2 < sinter
        @cast E[(l1, l2), (r1, r2)] := e11[l1, r1] * e21[l2, r1] * e12[l1, r2] * e22[l2, r2]
        return reshape(reshape(LE, (sbt, sl1 * sl2)) * E, (sb, st, sr1 * sr2))
    elseif sr1 <= sr2 && sl1 <= sl2
        LE = reshape(LE, sbt, sl1, 1, sl2) .* reshape(e21', 1, 1, sr1, sl2)  # [tb, l1, r1, l2]
        LE = reshape(reshape(LE, sbt * sl1 * sr1, sl2) * e22, (sbt, sl1, sr1, sr2))  # [tb, l1, r1, r2]
        LE .*= reshape(e11, 1, sl1, sr1, 1)  # [tb, l1, r1, r2] .* [:, l1, r1, :]
        LE .*= reshape(e12, 1, sl1, 1, sr2)  # [tb, l1, r1, r2] .* [:, l1, :, r2]
        LE = sum(LE, dims=2)
    elseif sr1 <= sr2 && sl2 <= sl1
        LE = permutedims(reshape(LE, (sbt, sl1, sl2)), (1, 3, 2))  # [tb, l2, l1]
        LE = reshape(LE, sbt, sl2, 1, sl1) .* reshape(e11', 1, 1, sr1, sl1)  # [tb, l2, r1, l1]
        LE = reshape(reshape(LE, sbt * sl2 * sr1, sl1) * e12, (sbt, sl2, sr1, sr2))  # [tb, l2, r1, r2]
        LE .*= reshape(e21, 1, sl2, sr1, 1)  # [tb, l2, r1, r2] .* [:, l2, r1, :]
        LE .*= reshape(e22, 1, sl2, 1, sr2)  # [tb, l2, r1, r2] .* [:, l2, :, r2]
        LE = sum(LE, dims=2)
    elseif sr2 <= sr1 && sl1 <= sl2
        LE = reshape(LE, sbt, sl1, 1, sl2) .* reshape(e22', 1, 1, sr2, sl2)  # [tb, l1, r2, l2]
        LE = reshape(reshape(LE, sbt * sl1 * sr2, sl2) * e21, (sbt, sl1, sr2, sr1))  # [tb, l1, r2, r1]
        LE .*= reshape(e11, 1, sl1, 1, sr1)  # [tb, l1, r2, r1] .* [:, l1, :, r1]
        LE .*= reshape(e12, 1, sl1, sr2, 1)  # [tb, l1, r2, r1] .* [:, l1, r2, :]
        LE = permutedims(dropdims(sum(LE, dims=2), dims=2), (1, 3, 2))
    else # sr2 <= sr1 && sl2 <= sl1
        LE = permutedims(reshape(LE, (sbt, sl1, sl2)), (1, 3, 2))  # [tb, l2, l1]
        LE = reshape(LE, sbt, sl2, 1, sl1) .* reshape(e12', 1, 1, sr2, sl1)  # [tb, l2, r2, l1]
        LE = reshape(reshape(LE, sbt * sl2 * sr2, sl1) * e11, (sbt, sl2, sr2, sr1))  # [tb, l2, r2, r1]
        LE .*= reshape(e21, 1, sl2, 1, sr1)  # [tb, l2, r2, r1] .* [:, l2, :, r1]
        LE .*= reshape(e22, 1, sl2, sr2, 1)  # [tb, l2, r2, r1] .* [:, l2, r2, :]
        LE = permutedims(dropdims(sum(LE, dims=2), dims=2), (1, 3, 2))
    end
    reshape(LE, sb, st, sr1 * sr2)
end