

function CUDA.CUSPARSE.CuSparseMatrixCSC(::Type{R}, pr::Vector{Int}) where R <: Real
    n = length(pr)
    CuSparseMatrixCSC(
        CuArray(1:n+1),
        CuArray(pr),
        CUDA.ones(R, n),
        (maximum(pr), n)
    )
end

function contract_sparse_with_three(X1, X2, X3, loc_exp, p1, p2, p3, pout)
    s1, s2, _ = size(X1)
    s3, s4, _ = size(X3)
    total_memory = 2^33
    batch_size = max(Int(floor(total_memory / (8 * (s1 * s2 + s2 * s3 + s3 * s4 + s4 * s1)))), 1)

    X1, X2, X3, loc_exp = CuArray.((X1, X2, X3, loc_exp))
    F = eltype(X1)
    out = CUDA.zeros(F, maximum(pout), s1, s4)

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
        ipr = CuSparseMatrixCSC(F, poutp .- (rf - 1))
        out[rf : rt, :, :] = out[rf : rt, :, :] .+ reshape(ipr * outp', (:, s1, s4))

        from = to + 1
    end
    Array(permutedims(out, (2, 1, 3)))
end

function update_env_left(
    LE::S, A::S, M::T, B::S, type::Union{Val{:n}, Val{:c}}
) where {S <: ArrayOrCuArray{3}, T <: SparseSiteTensor}
    B = permutedims(B, (3, 1, 2))
    LE = permutedims(LE, (1, 3, 2))
    A = permutedims(A, (1, 3, 2))
    order = type == Val{:n}() ? [4, 1, 2, 3] : [2, 1, 4, 3]
    contract_sparse_with_three(B, LE, A, M.loc_exp, M.projs[order]...)
end

function update_env_right(
    RE::S, A::S, M::SparseSiteTensor, B::S, type::Union{Val{:n}, Val{:c}}
) where S <: ArrayOrCuArray{3}
    A = permutedims(A, (1, 3, 2))
    RE = permutedims(RE, (1, 3, 2))
    B = permutedims(B, (3, 1, 2))
    order = type == Val{:n}() ? [2, 3, 4, 1] : [4, 3, 2, 1]
    contract_sparse_with_three(A, RE, B, M.loc_exp, M.projs[order]...)
end

function project_ket_on_bra(
    LE::S, B::S, M::SparseSiteTensor, RE::S, type::Union{Val{:n}, Val{:c}}
) where S <: ArrayOrCuArray{3}
    LE = permutedims(LE, (3, 1, 2))
    B = permutedims(B, (1, 3, 2))
    RE = permutedims(RE, (3, 1, 2))
    order = type == Val{:n}() ? [1, 4, 3, 2] : [1, 2, 3, 4]
    contract_sparse_with_three(LE, B, RE, M.loc_exp, M.projs[order]...)
end

# function cuda_max_batch_size(A::S, B::S) where {S <: ArrayOrCuArray{3}}
#     s1, s3 = size(A)[begin], size(A)[end]
#     s2, s4 = size(B)[begin], size(B)[end]
#     total_memory = 2^33
#     max(Int(floor(total_memory/(8 * (s1*s2 + s2*s3 + s3*s4 + s4*s1)))), 1)
# end

# function update_env_left(
#     LE::S, A::S, M::T, B::S, ::Val{:n}
# ) where {S <: ArrayOrCuArray{3}, T <: SparseSiteTensor}
#     F = eltype(LE)

#     total_size = length(M.projs[1])
#     batch_size = cuda_max_batch_size(A, B)

#     from = 1

#     L = CUDA.zeros(F, maximum(M.projs[3]), size(B, 3), size(A, 3))
#     while from <= total_size
#         to = min(total_size, from + batch_size - 1)

#         A_d = permutedims(CuArray(A[:, M.projs[2][from:to], :]), (1, 3, 2))
#         L_d = permutedims(CuArray(LE[:, M.projs[1][from:to], :]), (1, 3, 2))
#         B_d = permutedims(CuArray(B[:, M.projs[4][from:to], :]), (3, 1, 2))

#         Lr_d = B_d ⊠ L_d ⊠ A_d
#         Lr_d .*= reshape(CuArray(M.loc_exp[from:to]), 1, 1, :)
#         pr = M.projs[3][from:to]

#         ipr = CuSparseMatrixCSC(F, pr)
#         sb, st, _ = size(Lr_d)
#         @cast Lr_d[(x, y), z] := Lr_d[x, y, z]
#         L[1:maximum(pr), :, :] = L[1:maximum(pr), :, :] .+ reshape(ipr * Lr_d', (:, sb, st))
#         from = to + 1
#     end
#     Array(permutedims(L, (2, 1, 3)))
# end

# function update_env_left(
#     LE::S, A::S, M::T, B::S, ::Val{:c}
# ) where {S <: ArrayOrCuArray{3}, T <: SparseSiteTensor}
#     F = eltype(LE)

#     total_size = length(M.projs[1])
#     batch_size = cuda_max_batch_size(A, B)
#     from = 1

#     L = CUDA.zeros(F, maximum(M.projs[3]), size(B, 3), size(A, 3))
#     while from <= total_size
#         to = min(total_size, from + batch_size - 1)

#         A_d = permutedims(CuArray(A[:, M.projs[4][from:to], :]), (1, 3, 2))
#         L_d = permutedims(CuArray(LE[:, M.projs[1][from:to], :]), (1, 3, 2))
#         B_d = permutedims(CuArray(B[:, M.projs[2][from:to], :]), (3, 1, 2))

#         Lr_d = B_d ⊠ L_d ⊠ A_d
#         Lr_d .*= reshape(CuArray(M.loc_exp[from:to]), 1, 1, :)
#         pr = M.projs[3][from:to]

#         ipr = CuSparseMatrixCSC(F, pr)
#         sb, st, _ = size(Lr_d)
#         @cast Lr_d[(x, y), z] := Lr_d[x, y, z]
#         L[1:maximum(pr), :, :] = L[1:maximum(pr), :, :] .+ reshape(ipr * Lr_d', (:, sb, st))
#         from = to + 1
#     end
#     Array(permutedims(L, (2, 1, 3)))
# # end

# function update_env_right(
#     RE::S, A::S, M::SparseSiteTensor, B::S, ::Val{:n}
# ) where S <: ArrayOrCuArray{3}
#     F = eltype(RE)

#     total_size = length(M.projs[3])
#     batch_size = cuda_max_batch_size(A, B)
#     from = 1

#     R = CUDA.zeros(F,  maximum(M.projs[1]), size(A, 1), size(B, 1))
#     while from <= total_size
#         to = min(total_size, from + batch_size - 1)

#         A_d = permutedims(CuArray(A[:, M.projs[2][from:to], :]), (1, 3, 2))
#         R_d = permutedims(CuArray(RE[:, M.projs[3][from:to], :]), (1, 3, 2))
#         B_d = permutedims(CuArray(B[:, M.projs[4][from:to], :]), (3, 1, 2))

#         Rr_d = A_d ⊠ R_d ⊠ B_d
#         Rr_d .*= reshape(CuArray(M.loc_exp[from:to]), 1, 1, :)
#         pr = M.projs[1][from:to]

#         ipr = CuSparseMatrixCSC(F, pr)
#         sb, st, _ = size(Rr_d)
#         @cast Rr_d[(x, y), z] := Rr_d[x, y, z]
#         R[1:maximum(pr), :, :] = R[1:maximum(pr), :, :] .+ reshape(ipr * Rr_d', (:, sb, st))
#         from = to + 1
#     end
#     Array(permutedims(R, (2, 1, 3)))
# end

# function update_env_right(
#     RE::S, A::S, M::SparseSiteTensor, B::S, ::Val{:c}
# ) where S <: ArrayOrCuArray{3}
#     F = eltype(RE)

#     total_size = length(M.projs[3])
#     batch_size = cuda_max_batch_size(A, B)
#     from = 1

#     R = CUDA.zeros(F, maximum(M.projs[1]), size(A, 1), size(B, 1))
#     while from <= total_size
#         to = min(total_size, from + batch_size - 1)

#         A_d = permutedims(CuArray(A[:, M.projs[4][from:to], :]), (1, 3, 2))
#         R_d = permutedims(CuArray(RE[:, M.projs[3][from:to], :]), (1, 3, 2))
#         B_d = permutedims(CuArray(B[:, M.projs[2][from:to], :]), (3, 1, 2))

#         Rr_d = A_d ⊠ R_d ⊠ B_d
#         Rr_d .*= reshape(CuArray(M.loc_exp[from:to]), 1, 1, :)
#         pr = M.projs[1][from:to]

#         ipr = CuSparseMatrixCSC(F, pr)
#         sb, st, _ = size(Rr_d)
#         @cast Rr_d[(x, y), z] := Rr_d[x, y, z]
#         R[1:maximum(pr), :, :] = R[1:maximum(pr), :, :] .+ reshape(ipr * Rr_d', (:, sb, st))
#         from = to + 1
#     end
#     Array(permutedims(R, (2, 1, 3)))
# end

# function project_ket_on_bra(
#     LE::S, B::S, M::SparseSiteTensor, RE::S, ::Val{:n}
# ) where S <: ArrayOrCuArray{3}
#     F = eltype(LE)

#     total_size = length(M.projs[3])
#     batch_size = cuda_max_batch_size(LE, RE)
#     from = 1

#     A = CUDA.zeros(F, maximum(M.projs[2]), size(LE, 3), size(RE, 1))
#     while from <= total_size
#         to = min(total_size, from + batch_size - 1)

#         le = permutedims(CuArray(LE[:, M.projs[1][from:to], :]), (3, 1, 2))
#         b = permutedims(CuArray(B[:, M.projs[4][from:to], :]), (1, 3, 2))
#         re = permutedims(CuArray(RE[:, M.projs[3][from:to], :]), (3, 1, 2))

#         Ar_d = le ⊠ b ⊠ re
#         Ar_d .*= reshape(CuArray(M.loc_exp[from:to]), 1, 1, :)
#         pu = M.projs[2][from:to]

#         ipu = CuSparseMatrixCSC(F, pu)
#         sb, st, _ = size(Ar_d)
#         @cast Ar_d[(x, y), z] := Ar_d[x, y, z]
#         A[1:maximum(pu), :, :] = A[1:maximum(pu), :, :] .+ reshape(ipu * Ar_d', (:, sb, st))
#         from = to + 1
#     end
#     Array(permutedims(A, (2, 1, 3)))
# end

# function project_ket_on_bra(
#     LE::S, B::S, M::SparseSiteTensor, RE::S, ::Val{:c}
# ) where S <: ArrayOrCuArray{3}
#     F = eltype(LE)

#     total_size = length(M.projs[3])
#     batch_size = cuda_max_batch_size(LE, RE)
#     from = 1

#     A = CUDA.zeros(F, maximum(M.projs[4]), size(LE, 3), size(RE, 1))
#     while from <= total_size
#         to = min(total_size, from + batch_size - 1)

#         le = permutedims(CuArray(LE[:, M.projs[1][from:to], :]), (3, 1, 2))
#         b = permutedims(CuArray(B[:, M.projs[2][from:to], :]), (1, 3, 2))
#         re = permutedims(CuArray(RE[:, M.projs[3][from:to], :]), (3, 1, 2))

#         Ar_d = le ⊠ b ⊠ re
#         Ar_d .*= reshape(CuArray(M.loc_exp[from:to]), 1, 1, :)
#         pu = M.projs[4][from:to]

#         ipu = CuSparseMatrixCSC(F, pu)
#         sb, st, _ = size(Ar_d)
#         @cast Ar_d[(x, y), z] := Ar_d[x, y, z]
#         A[1:maximum(pu), :, :] = A[1:maximum(pu), :, :] .+ reshape(ipu * Ar_d', (:, sb, st))
#         from = to + 1
#     end
#     Array(permutedims(A, (2, 1, 3)))
# end
