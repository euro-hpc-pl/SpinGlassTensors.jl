function contract_sparse_with_three(
        X1::S, X2::S, X3::S, loc_exp::T, p1::Q, p2::Q, p3::Q, pout::Q
) where {S <: CuArray{R, 3}, T <: CuArray{R, 1}, Q <: Array{Int, 1}} where R <: Real
    s1, s2, _ = size(X1)
    s3, s4, _ = size(X3)

    #TODO add better handling for this
    total_memory = 2^33
    batch_size = max(Int(floor(total_memory / (8 * (s1 * s2 + s2 * s3 + s3 * s4 + s4 * s1)))), 1)

    out = CUDA.zeros(R, maximum(pout), s1, s4)
    from, total_size = 1, length(p1)
    while from <= total_size
        to = min(total_size, from + batch_size - 1)
        X1p = X1[:, :, p1[from:to]]
        X2p = X2[:, :, p2[from:to]]
        X3p = X3[:, :, p3[from:to]]
        outp = X1p ⊠ X2p ⊠ X3p
        outp .*= reshape(loc_exp[from:to], 1, 1, :)
        @cast outp[(x, y), z] := outp[x, y, z]
        poutp = pout[from:to]
        rf, rt = minimum(poutp), maximum(poutp)
        ipr = CuSparseMatrixCSC(R, poutp .- (rf - 1))
        out[rf:rt, :, :] = out[rf:rt, :, :] .+ reshape(ipr * outp', (:, s1, s4))
        from = to + 1
    end
    permutedims(out, (2, 1, 3))
end

function update_env_left(LE::S, A::S, M::T, B::S) where {S <: CuArray{R, 3}, T <: SiteTensor{R}} where R <: Real
    contract_sparse_with_three(
        permutedims(B, (3, 1, 2)),
        permutedims(LE, (1, 3, 2)),
        permutedims(A, (1, 3, 2)),
        M.loc_exp,
        M.projs[[4, 1, 2, 3]]...
    )
end

function update_env_right(RE::S, A::S, M::SiteTensor{R}, B::S) where {S <: CuArray{R, 3}} where R <: Real
    contract_sparse_with_three(
        permutedims(A, (1, 3, 2)),
        permutedims(RE, (1, 3, 2)),
        permutedims(B, (3, 1, 2)),
        M.loc_exp,
        M.projs[[2, 3, 4, 1]]...
    )
end

function project_ket_on_bra(LE::S, B::S, M::SiteTensor{R}, RE::S) where {S <: CuArray{R, 3}} where R <: Real
    contract_sparse_with_three(
        permutedims(LE, (3, 1, 2)),
        permutedims(B, (1, 3, 2)),
        permutedims(RE, (3, 1, 2)),
        M.loc_exp,
        M.projs[[1, 4, 3, 2]]...
    )
end

function update_reduced_env_right(K::CuArray{T, 1}, RE::CuArray{T, 2}, M::SiteTensor{T}, B::CuArray{T, 3}) where T <: Real
    Bp = permutedims(B, (1, 3, 2))[:, :, M.projs[4]]
    REp = reshape(RE, (size(RE, 1), 1, size(RE, 2)))[:, :, M.projs[3]]
    outp = dropdims(Bp ⊠ REp, dims=2) .* reshape(M.loc_exp .* K[M.projs[2]], 1, :)
    permutedims(CuSparseMatrixCSC(T, M.projs[1]) * outp', (2, 1))
end

function contract_tensors43(M::SiteTensor{T, 4}, B::CuArray{T, 3}) where T <: Real
    sb1, _, sb3 = size(B)
    sm1, sm2, sm3 = maximum.(M.projs[1:3])
    Bp = permutedims(B, (1, 3, 2))[:, :, M.projs[4]] .* reshape(M.loc_exp, 1, 1, :)
    @cast Bp[(x, y), z] := Bp[x, y, z]
    p123 = M.projs[1] .+ (M.projs[2] .- 1) .* sm1 .+ (M.projs[3] .- 1) .* (sm1 * sm2)
    ip123 = CuSparseMatrixCSC(eltype(B), p123)
    out = reshape(ip123 * Bp', (sm1, sm2, sm3, sb1, sb3))
    reshape(permutedims(out, (4, 1, 2, 5, 3)), sb1 * sm1, sm2, sb3 * sm3)
end

function corner_matrix(C::S, M::T, B::S) where {S <: CuArray{R, 3}, T <: SiteTensor{R, 4}} where R <: Real
    Bp = permutedims(B, (1, 3, 2))[:, :, M.projs[4]]
    Cp = permutedims(C, (3, 1, 2))[:, :, M.projs[3]]
    outp = Bp ⊠ Cp .* reshape(M.loc_exp, 1, 1, :)
    @cast outp[(x, y), z] := outp[x, y, z]
    sm1 = maximum(M.projs[1])
    p12 = M.projs[1] .+ (M.projs[2] .- 1) .* sm1
    ip12 = CuSparseMatrixCSC(R, p12)
    out = reshape(ip12 * outp', (sm1, maximum(M.projs[2]), size(B, 1), size(C, 1)))
    permutedims(out, (3, 1, 2, 4))
end
