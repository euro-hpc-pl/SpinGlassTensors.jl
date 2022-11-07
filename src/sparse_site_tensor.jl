"""
$(TYPEDSIGNATURES)
"""
function update_env_left(
    LE::S, A::S, M::T, B::S, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: SparseSiteTensor}
# @time begin
    #println("update_env_left   SparseSiteTensor")
    total_size = length(M.projs[1])
    batch_size = min(2^20, total_size)
    from = 1

    L = CUDA.zeros(eltype(LE),  maximum(M.projs[3]), size(B, 3), size(A, 3))
    while from <= total_size
        to = min(total_size, from + batch_size - 1)

        A_d = permutedims(CUDA.CuArray(A[:, M.projs[2][from : to], :]), (1, 3, 2))
        L_d = permutedims(CUDA.CuArray(LE[:, M.projs[1][from : to], :]), (1, 3, 2))
        B_d = permutedims(CUDA.CuArray(B[:, M.projs[4][from : to], :]), (3, 1, 2))

        Lr_d = B_d ⊠ L_d ⊠ A_d
        Lr_d .*= reshape(CUDA.CuArray(M.loc_exp[from:to]), 1, 1, :)
        pr = M.projs[3][from:to]

        ipr = _cusparse_projector(pr)
        sb, st, _ = size(Lr_d)
        @cast Lr_d[(x, y), z] := Lr_d[x, y, z]
        L[1:maximum(pr), :, :] = L[1:maximum(pr), :, :] .+ reshape(ipr * Lr_d', (:, sb, st))
        from = to + 1
    end
# end
    Array(permutedims(L, (2, 1, 3)))
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_left(
    LE::S, A::S, M::T, B::S, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: SparseSiteTensor}
    total_size = length(M.projs[1])
    batch_size = min(2^20, total_size)
    from = 1

    L = CUDA.zeros(eltype(LE), maximum(M.projs[3]), size(B, 3), size(A, 3))
    while from <= total_size
        to = min(total_size, from + batch_size - 1)

        A_d = permutedims(CUDA.CuArray(A[:, M.projs[4][from : to], :]), (1, 3, 2))
        L_d = permutedims(CUDA.CuArray(LE[:, M.projs[1][from : to], :]), (1, 3, 2))
        B_d = permutedims(CUDA.CuArray(B[:, M.projs[2][from : to], :]), (3, 1, 2))

        Lr_d = B_d ⊠ L_d ⊠ A_d
        Lr_d .*= reshape(CUDA.CuArray(M.loc_exp[from:to]), 1, 1, :)
        pr = M.projs[3][from:to]

        ipr = _cusparse_projector(pr)
        sb, st, _ = size(Lr_d)
        @cast Lr_d[(x, y), z] := Lr_d[x, y, z]
        L[1:maximum(pr), :, :] = L[1:maximum(pr), :, :] .+ reshape(ipr * Lr_d', (:, sb, st))
        from = to + 1
    end
    Array(permutedims(L, (2, 1, 3)))
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_right(
    RE::S, A::S, M::T, B::S, ::Val{:n}
) where {T <: SparseSiteTensor, S <: AbstractArray{Float64, 3}}
    total_size = length(M.projs[3])
    batch_size = min(2^24, total_size)
    from = 1

    R = CUDA.zeros(eltype(RE),  maximum(M.projs[1]), size(A, 1), size(B, 1))
    while from <= total_size
        to = min(total_size, from + batch_size - 1)

        A_d = permutedims(CUDA.CuArray(A[:, M.projs[2], :]), (1, 3, 2))
        R_d = permutedims(CUDA.CuArray(RE[:, M.projs[3], :]), (1, 3, 2))
        B_d = permutedims(CUDA.CuArray(B[:, M.projs[4], :]), (3, 1, 2))
    
        Rr_d = A_d ⊠ R_d ⊠ B_d
        Rr_d .*= reshape(CUDA.CuArray(M.loc_exp[from:to]), 1, 1, :)
        pr = M.projs[1][from:to]

        ipr = _cusparse_projector(pr)
        sb, st, _ = size(Rr_d)
        @cast Rr_d[(x, y), z] := Rr_d[x, y, z]
        R[1:maximum(pr), :, :] = R[1:maximum(pr), :, :] .+ reshape(ipr * Rr_d', (:, sb, st))
        from = to + 1
    end
    Array(permutedims(R, (2, 1, 3)))
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_right(
    RE::S, A::S, M::T, B::S, ::Val{:c}
) where {T <: SparseSiteTensor, S <: AbstractArray{Float64, 3}}
    total_size = length(M.projs[3])
    batch_size = min(2^24, total_size)
    from = 1

    R = CUDA.zeros(eltype(RE),  maximum(M.projs[1]), size(A, 1), size(B, 1))
    while from <= total_size
        to = min(total_size, from + batch_size - 1)

        A_d = permutedims(CUDA.CuArray(A[:, M.projs[4], :]), (1, 3, 2))
        R_d = permutedims(CUDA.CuArray(RE[:, M.projs[3], :]), (1, 3, 2))
        B_d = permutedims(CUDA.CuArray(B[:, M.projs[2], :]), (3, 1, 2))
    
        Rr_d = A_d ⊠ R_d ⊠ B_d
        Rr_d .*= reshape(CUDA.CuArray(M.loc_exp[from:to]), 1, 1, :)
        pr = M.projs[1][from:to]

        ipr = _cusparse_projector(pr)
        sb, st, _ = size(Rr_d)
        @cast Rr_d[(x, y), z] := Rr_d[x, y, z]
        R[1:maximum(pr), :, :] = R[1:maximum(pr), :, :] .+ reshape(ipr * Rr_d', (:, sb, st))
        from = to + 1
    end
    Array(permutedims(R, (2, 1, 3)))
end

"""
$(TYPEDSIGNATURES)
"""
function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: SparseSiteTensor}
    total_size = length(M.projs[3])
    batch_size = min(2^24, total_size)
    from = 1

    A = CUDA.zeros(eltype(LE), maximum(M.projs[2]), size(LE, 3), size(RE, 1))
    while from <= total_size
        to = min(total_size, from + batch_size - 1)

        le = permutedims(CUDA.CuArray(LE[:, M.projs[1], :]), (3, 1, 2))
        b = permutedims(CUDA.CuArray(B[:, M.projs[4], :]), (1, 3, 2))
        re = permutedims(CUDA.CuArray(RE[:, M.projs[3], :]), (3, 1, 2))
    
        Ar_d = le ⊠ b ⊠ re
        Ar_d .*= reshape(CUDA.CuArray(M.loc_exp[from:to]), 1, 1, :)
        pu = M.projs[2][from:to]

        ipu = _cusparse_projector(pu)
        sb, st, _ = size(Ar_d)
        @cast Ar_d[(x, y), z] := Ar_d[x, y, z]
        A[1:maximum(pu), :, :] = A[1:maximum(pu), :, :] .+ reshape(ipu * Ar_d', (:, sb, st))
        from = to + 1
    end
    Array(permutedims(A, (2, 1, 3)))
end

"""
$(TYPEDSIGNATURES)
"""
function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: SparseSiteTensor}
    total_size = length(M.projs[3])
    batch_size = min(2^24, total_size)
    from = 1

    A = CUDA.zeros(eltype(LE), maximum(M.projs[4]), size(LE, 3), size(RE, 1))
    while from <= total_size
        to = min(total_size, from + batch_size - 1)

        le = permutedims(CUDA.CuArray(LE[:, M.projs[1], :]), (3, 1, 2))
        b = permutedims(CUDA.CuArray(B[:, M.projs[2], :]), (1, 3, 2))
        re = permutedims(CUDA.CuArray(RE[:, M.projs[3], :]), (3, 1, 2))
    
        Ar_d = le ⊠ b ⊠ re
        Ar_d .*= reshape(CUDA.CuArray(M.loc_exp[from:to]), 1, 1, :)
        pu = M.projs[4][from:to]

        ipu = _cusparse_projector(pu)
        sb, st, _ = size(Ar_d)
        @cast Ar_d[(x, y), z] := Ar_d[x, y, z]
        A[1:maximum(pu), :, :] = A[1:maximum(pu), :, :] .+ reshape(ipu * Ar_d', (:, sb, st))
        from = to + 1
    end
    Array(permutedims(A, (2, 1, 3)))
end

function _cusparse_projector(pr::Vector{Int})
    # This is how sparse matrix is represented internally
    csrRowPtr = CuArray(collect(1:length(pr) + 1))
    csrColInd = CuArray(pr)
    csrNzVal = CUDA.ones(Float64, length(pr))
    ipr = CUSPARSE.CuSparseMatrixCSC(csrRowPtr, csrColInd, csrNzVal, (maximum(pr), length(pr))) # transposed right here
end

# function update_env_left(
#     LE::S, A::S, M::T, B::S, ::Val{:c}
# ) where {S <: AbstractArray{Float64, 3}, T <: SparseSiteTensor}

#     L = CUDA.zeros(eltype(LE), maximum(M.projs[3]), size(B, 3), size(A, 3))

#     A_d = permutedims(CUDA.CuArray(A[:, M.projs[4], :]), (1, 3, 2))
#     L_d = permutedims(CUDA.CuArray(LE[:, M.projs[1], :]), (1, 3, 2))
#     B_d = permutedims(CUDA.CuArray(B[:, M.projs[2], :]), (3, 1, 2))

#     Lr_d = B_d ⊠ L_d ⊠ A_d
#     Lr_d .*= reshape(CUDA.CuArray(M.loc_exp), 1, 1, :)

#     pr = M.projs[3]

#     ipr = _cusparse_projector(pr)

#     Lr_d = permutedims(Lr_d, (3, 2, 1))
#     _, sy, sz = size(Lr_d)
#     @cast Lr_d[x, (y, z)] := Lr_d[x, y, z]

#     L = ipr * Lr_d
#     L = reshape(L, (:, sy, sz))

#     Array(permutedims(L, (3, 1, 2)) ./ maximum(abs.(L)))
# end

# """
# $(TYPEDSIGNATURES)
# """
# function update_env_right(
#     RE::S, A::S, M::T, B::S, ::Val{:n}
# ) where {T <: SparseSiteTensor, S <: AbstractArray{Float64, 3}}
# # @time begin
# #     println("update_env_right   SparseSiteTensor")
#     R = CUDA.zeros(eltype(RE), size(A, 1), size(B, 1), maximum(M.projs[1]))

#     A_d = permutedims(CUDA.CuArray(A[:, M.projs[2], :]), (1, 3, 2))
#     R_d = permutedims(CUDA.CuArray(RE[:, M.projs[3], :]), (1, 3, 2))
#     B_d = permutedims(CUDA.CuArray(B[:, M.projs[4], :]), (3, 1, 2))

#     Rr_d = A_d ⊠ R_d ⊠ B_d

#     Rr_d .*= reshape(CUDA.CuArray(M.loc_exp), 1, 1, :)
#     pr = M.projs[1]

#     ipr = _cusparse_projector(pr)

#     Rr_d = permutedims(Rr_d, (3, 2, 1))
#     _, sy, sz = size(Rr_d)
#     @cast Rr_d[x, (y, z)] := Rr_d[x, y, z]

#     R = ipr * Rr_d
#     R = reshape(R, (:, sy, sz))
# # end
#     Array(permutedims(R, (3, 1, 2)))
# end

# """
# $(TYPEDSIGNATURES)
# """
# function update_env_right(
#     RE::S, A::S, M::T, B::S, ::Val{:c}
# ) where {T <: SparseSiteTensor, S <: AbstractArray{Float64, 3}}
#     R = CUDA.zeros(eltype(RE), size(A, 1), size(B, 1), maximum(M.projs[1]))

#     A_d = permutedims(CUDA.CuArray(A[:, M.projs[4], :]), (1, 3, 2))
#     R_d = permutedims(CUDA.CuArray(RE[:, M.projs[3], :]), (1, 3, 2))
#     B_d = permutedims(CUDA.CuArray(B[:, M.projs[2], :]), (3, 1, 2))

#     Rr_d = A_d ⊠ R_d ⊠ B_d

#     Rr_d .*= reshape(CUDA.CuArray(M.loc_exp), 1, 1, :)
#     pr = M.projs[1]

#     ipr = _cusparse_projector(pr)

#     Rr_d = permutedims(Rr_d, (3, 2, 1)) #(256, 4, 4)
#     _, sy, sz = size(Rr_d)
#     @cast Rr_d[x, (y, z)] := Rr_d[x, y, z]

#     R = ipr * Rr_d  #(16, 16)
#     R = reshape(R, (:, sy, sz))

#     Array(permutedims(R, (3, 1, 2)))
# end

# """
# $(TYPEDSIGNATURES)
# """
# function project_ket_on_bra(
#     LE::S, B::S, M::T, RE::S, ::Val{:n}
# ) where {S <: AbstractArray{Float64, 3}, T <: SparseSiteTensor}

# # @time begin
# #     println("project_ket_on_bra   SparseSiteTensor")
#     A = CUDA.zeros(eltype(LE), size(LE, 3), size(RE, 1), maximum(M.projs[2]))

#     le = permutedims(CUDA.CuArray(LE[:, M.projs[1], :]), (3, 1, 2))
#     b = permutedims(CUDA.CuArray(B[:, M.projs[4], :]), (1, 3, 2))
#     re = permutedims(CUDA.CuArray(RE[:, M.projs[3], :]), (3, 1, 2))

#     Ar_d = le ⊠ b ⊠ re
#     Ar_d .*= reshape(CUDA.CuArray(M.loc_exp), 1, 1, :)

#     pu = M.projs[2]

#     ipu = _cusparse_projector(pu)

#     Ar_d = permutedims(Ar_d, (3, 2, 1)) #(256, 4, 4)
#     _, sy, sz = size(Ar_d)
#     @cast Ar_d[x, (y, z)] := Ar_d[x, y, z]

#     A = ipu * Ar_d  #(16, 16)
#     A = reshape(A, (:, sy, sz))
# # end
#     Array(permutedims(A, (3, 1, 2)))
# end

# """
# $(TYPEDSIGNATURES)
# """
# function project_ket_on_bra(
#     LE::S, B::S, M::T, RE::S, ::Val{:c}
# ) where {S <: AbstractArray{Float64, 3}, T <: SparseSiteTensor}
# A = CUDA.zeros(eltype(LE), size(LE, 3), size(RE, 1), maximum(M.projs[4]))

# le = permutedims(CUDA.CuArray(LE[:, M.projs[1], :]), (3, 1, 2))
# b = permutedims(CUDA.CuArray(B[:, M.projs[2], :]), (1, 3, 2))
# re = permutedims(CUDA.CuArray(RE[:, M.projs[3], :]), (3, 1, 2))

# Ar_d = le ⊠ b ⊠ re
# Ar_d .*= reshape(CUDA.CuArray(M.loc_exp), 1, 1, :)

# pu = M.projs[4]

# for i in 1:maximum(pu)
#     A[:,:,i] = sum(Ar_d[:, :, pu.==i], dims=3)
# end
# Array(permutedims(A, (1, 3, 2)))
# end