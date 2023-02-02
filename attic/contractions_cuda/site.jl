# #TODO make sure slicing is done right, cf. https://discourse.julialang.org/t/correct-implementation-of-cuarrays-slicing-operations/90600

# function contract_sparse_with_three(
#         X1::S, X2::S, X3::S, loc_exp::T, p1::Q, p2::Q, p3::Q, pout::Q
# ) where {S <: CuArray{R, 3}, T <: CuArray{R, 1}, Q <: CuArray{Int, 1}} where R <: Real
#     s1, s2, _ = size(X1)
#     s3, s4, _ = size(X3)

#     #TODO add better handling for this
#     total_memory = 2^33
#     batch_size = max(Int(floor(total_memory / (8 * (s1 * s2 + s2 * s3 + s3 * s4 + s4 * s1)))), 1)

#     out = CUDA.zeros(R, maximum(pout), s1, s4)
#     from = 1
#     total_size = length(p1)
#     while from <= total_size
#         to = min(total_size, from + batch_size - 1)
#         @inbounds X1p = X1[:, :, p1[from:to]]
#         @inbounds X2p = X2[:, :, p2[from:to]]
#         @inbounds X3p = X3[:, :, p3[from:to]]

#         outp = X1p ⊠ X2p ⊠ X3p
#         le = @view loc_exp[from:to]
#         outp .*= reshape(le, 1, 1, :)
#         @cast outp[(x, y), z] := outp[x, y, z]

#         poutp = @view pout[from:to]
#         rf = minimum(poutp)
#         rt = maximum(poutp)
#         ipr = CuSparseMatrixCSC(R, poutp .- (rf - 1)) #  TODO take it out from this loop (?)
#         @inbounds out[rf:rt, :, :] .+= reshape(ipr * outp', :, s1, s4)

#         from = to + 1
#         CUDA.unsafe_free!.((X1p, X3p, X3p, outp))
#     end
#     permutedims(out, (2, 3, 1))
# end

# function update_env_left(LE::S, A::S, M::T, B::S) where {S <: CuArray{R, 3}, T <: SiteTensor{R}} where R <: Real
#     contract_sparse_with_three(permutedims(B, (2, 1, 3)), LE, A, M.loc_exp, M.projs[[4, 1, 2, 3]]...)
# end

# function update_env_right(RE::S, A::S, M::SiteTensor{R}, B::S) where {S <: CuArray{R, 3}} where R <: Real
#     contract_sparse_with_three(B, RE, permutedims(A, (2, 1, 3)), M.loc_exp, M.projs[[4, 3, 2, 1]]...)
# end

# function project_ket_on_bra(LE::S, B::S, M::SiteTensor{R}, RE::S) where {S <: CuArray{R, 3}} where R <: Real
#     contract_sparse_with_three(permutedims(LE, (2, 1, 3)), B, RE, M.loc_exp, M.projs[[1, 4, 3, 2]]...)
# end

# function update_reduced_env_right(K::CuArray{T, 1}, RE::CuArray{T, 2}, M::SiteTensor{T}, B::CuArray{T, 3}) where T <: Real
#     @inbounds Bp = B[:, :, M.projs[4]]
#     REp = reshape(RE, size(RE, 1), 1, size(RE, 2))
#     @inbounds REp = REp[:, :, M.projs[3]]
#     outp = dropdims(Bp ⊠ REp, dims=2) .* reshape(M.loc_exp .* K[M.projs[2]], 1, :)
#     ipr = CuSparseMatrixCSC(T, M.projs[1])
#     permutedims(ipr * outp', (2, 1))
# end

# function contract_tensors43(M::SiteTensor{T, 4}, B::CuArray{T, 3}) where T <: Real
#     sb1, sb2, _ = size(B)
#     sm1, sm2, sm3 = maximum.(M.projs[1:3])
#     @inbounds Bp = B[:, :, M.projs[4]] .* reshape(M.loc_exp, 1, 1, :)
#     @cast Bp[(x, y), z] := Bp[x, y, z]
#     ip123 = CuSparseMatrixCSC(T, M.projs[1], M.projs[2], M.projs[3])
#     out = reshape(ip123 * Bp', sm1, sm2, sm3, sb1, sb2)
#     out = permutedims(out, (4, 1, 5, 3, 2))
#     reshape(out, sb1 * sm1, sb2 * sm3, sm2)
# end

# function corner_matrix(C::S, M::T, B::S) where {S <: CuArray{R, 3}, T <: SiteTensor{R, 4}} where R <: Real
#     @inbounds Bp = B[:, :, M.projs[4]]
#     @inbounds Cp = C[:, :, M.projs[3]]
#     outp = Bp ⊠ Cp
#     outp .*= reshape(M.loc_exp, 1, 1, :)
#     @cast outp[(x, y), z] := outp[x, y, z]
#     sm1 = maximum(M.projs[1])
#     @inbounds p12 = M.projs[1] .+ (M.projs[2] .- 1) .* sm1
#     ip12 = CuSparseMatrixCSC(R, p12)
#     out = reshape(ip12 * outp', sm1, maximum(M.projs[2]), size(B, 1), size(C, 2))
#     permutedims(out, (3, 1, 4, 2))
# end
