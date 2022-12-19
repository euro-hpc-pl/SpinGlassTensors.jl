
#TODO this function is only for now
function _batch_size(s1::T, s2::T, s3::T, s4::T) where T <: Int
    total_memory = 2 ^ 33
    max(Int(floor(total_memory / (8 * (s1 * s2 + s2 * s3 + s3 * s4 + s4 * s1)))), 1)
end

function contract_sparse_with_three(
    X_1::T, X_2::T, X_3::T, loc_exp::Array{R, 1}, p_1, p_2, p_3, pout
) where {T <: ArrayOrCuArray{R, 3} where R <: Real}
    s1, s2, _, s3, s4, _  = size(X_1), size(X_3)

    @nexprs 3 k -> X_k = CuArray(X_k)
    loc_exp = CuArray(loc_exp)

    from = 1
    total_size = length(p1)
    batch_size = _batch_size(s1, s2, s3, s4)

    out = CUDA.zeros(R, maximum(pout), s1, s4)
    while from <= total_size
        to = min(total_size, from + batch_size - 1)
        @nexprs 3 k -> Xp_k = X_k[:, :, p_k[from:to]]
        outp = Xp_1 ⊠ Xp_2 ⊠ Xp_3
        outp .*= reshape(loc_exp[from:to], 1, 1, :)
        @cast outp[(x, y), z] := outp[x, y, z]
        poutp = pout[from:to]
        rf, rt = minimum(poutp), maximum(poutp)
        ipr = CuSparseMatrixCSC(R, poutp .- (rf - 1))
        out[rf:rt, :, :] .+= reshape(ipr * outp', (:, s1, s4))
        from = to + 1
    end
    permutedims(out, (2, 1, 3)) |> Array
end

function update_env_left(
    LE::S, A::S, M::T, B::S
) where {S <: ArrayOrCuArray{R, 3}, T <: SparseSiteTensor{R} where R <: Real}
    B = permutedims(B, (3, 1, 2))
    LE = permutedims(LE, (1, 3, 2))
    A = permutedims(A, (1, 3, 2))
    Mp = M.projs[[2, 1, 4, 3]]
    contract_sparse_with_three(B, LE, A, M.loc_exp, Mp...)
end

function update_env_right(
    RE::S, A::S, M::SparseSiteTensor{R}, B::S
) where {S <: ArrayOrCuArray{R, 3} where R <: Real}
    A = permutedims(A, (1, 3, 2))
    RE = permutedims(RE, (1, 3, 2))
    B = permutedims(B, (3, 1, 2))
    Mp = M.projs[[4, 3, 2, 1]]
    contract_sparse_with_three(A, RE, B, M.loc_exp, Mp...)
end

function project_ket_on_bra(
    LE::S, B::S, M::SparseSiteTensor{R}, RE::S
) where {S <: ArrayOrCuArray{R, 3} where R <: Real}
    LE = permutedims(LE, (3, 1, 2))
    B = permutedims(B, (1, 3, 2))
    RE = permutedims(RE, (3, 1, 2))
    Mp = M.projs[[1, 2, 3, 4]]
    contract_sparse_with_three(LE, B, RE, M.loc_exp, Mp...)
end

function update_reduced_env_right(
    K::Array{T, 1}, RE::Array{T, 2}, M::SparseSiteTensor{T}, B::Array{T, 3} # TODO can B, RE, M.loc_exp, K be put on GPU already?
) where T <: Real
    B, RE, loc_exp, K = CuArray.((B, RE, M.loc_exp, K))

    Bp = permutedims(B, (1, 3, 2))[:, :, M.projs[4]]
    RE = reshape(RE, (size(RE, 1), 1, size(RE, 2)))
    REp = RE[:, :, M.projs[3]]

    outp = dropdims(Bp ⊠ REp, dims=2)
    outp .*= reshape(loc_exp .* K[M.projs[2]], 1, :)
    R = CuSparseMatrixCSC(T, M.projs[1]) * outp'
    Array(R')
end
