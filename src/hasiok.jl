# Improve this functiion with brodcasting
# function update_env_left(
#     LE::S, A::S, M::T, B::S, ::Val{:n}
# ) where {S <: AbstractArray{Float64, 3}, T <: SparseSiteTensor}
#     L = zeros(size(B, 3), maximum(M.projs[3]), size(A, 3))
#     for (σ, lexp) ∈ enumerate(M.loc_exp)
#         AA = @view A[:, M.projs[2][σ], :]
#         LL = @view LE[:, M.projs[1][σ], :]
#         BB = @view B[:, M.projs[4][σ], :]
#         L[:, M.projs[3][σ], :] += lexp .* (BB' * LL * AA)
#     end
#     L
# end

# function update_env_left(
#     LE::S, A::S, M::T, B::S, ::Val{:c}
# ) where {S <: AbstractArray{Float64, 3}, T <: SparseSiteTensor}
#     L = zeros(size(B, 3), maximum(M.projs[3]), size(A, 3))

#     for (σ, lexp) ∈ enumerate(M.loc_exp)
#         AA = @view A[:, M.projs[4][σ], :]
#         LL = @view LE[:, M.projs[1][σ], :]
#         BB = @view B[:, M.projs[2][σ], :]
#         L[:, M.projs[3][σ], :] += lexp .* (BB' * LL * AA)
#     end
#     L
# end

# # This is not optimal
# function update_env_left(
#     LE::S, A::S, M::T, B::S, ::Val{:n}
# ) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}
#     h = M.con
#     p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

#     @cast A4[x, k, l, y] := A[x, (k, l), y] (k ∈ 1:maximum(p_rt))
#     @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lb))

#     L = zeros(size(B, 3), length(p_r), size(A, 3))
#     for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
#         AA = @view A4[:, p_rt[r], p_lt[l], :]
#         LL = @view LE[:, l, :]
#         BB = @view B4[:, p_lb[l], p_rb[r], :]
#         L[:, r, :] += h[p_l[l], p_r[r]] .* (BB' * LL * AA)
#     end
#     L
# end

# function update_env_left(
#     LE::S, A::S, M::T, B::S, ::Val{:c}
# ) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}
#     ## TO BE WRITTEN
#     h = M.con
#     p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

#     @cast A4[x, k, l, y] := A[x, (k, l), y] (k ∈ 1:maximum(p_rt))
#     @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lb))

#     L = zeros(size(B, 3), length(p_r), size(A, 3))
#     for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
#         AA = @view A4[:, p_lb[l], p_rb[r], :]
#         LL = @view LE[:, l, :]
#         BB = @view B4[:, p_rt[r], p_lt[l], :]
#         L[:, r, :] += h[p_l[l], p_r[r]] .* (BB' * LL * AA)
#     end
#     L
# end

# function update_env_right(
#     RE::S, A::S, M::T, B::S, ::Val{:n}
# ) where {T <: SparseSiteTensor, S} # {T <: SparseSiteTensor, S <: AbstractArray{Float64, 3}}
#     R = zeros(size(A, 1), maximum(M.projs[1]), size(B, 1))

#     #Threads.@threads for σ ∈ 1:length(M.loc_exp)
#     #    lexp = M.loc_exp[σ]
#     for (σ, lexp) ∈ enumerate(M.loc_exp)
#         AA = @view A[:, M.projs[2][σ], :]
#         RR = @view RE[:, M.projs[3][σ], :]
#         BB = @view B[:, M.projs[4][σ], :]
#         R[:, M.projs[1][σ], :] += lexp .* (AA * RR * BB')
#     end
#     R
# end

# function update_env_right(
#     RE::S, A::S, M::T, B::S, ::Val{:c}
# ) where {T <: SparseSiteTensor, S <: AbstractArray{Float64, 3}}
#     R = zeros(size(A, 1), maximum(M.projs[1]), size(B, 1))

#     #Threads.@threads for σ ∈ 1:length(M.loc_exp)
#     #    lexp = M.loc_exp[σ]
#     for (σ, lexp) ∈ enumerate(M.loc_exp)
#         AA = @view A[:, M.projs[4][σ], :]
#         RR = @view RE[:, M.projs[3][σ], :]
#         BB = @view B[:, M.projs[2][σ], :]
#         R[:, M.projs[1][σ], :] += lexp .* (AA * RR * BB')
#     end
#     R
# end

# function update_env_right(
#     RE::S, A::S, M::T, B::S, ::Val{:n}
# ) where {T <: SparseVirtualTensor, S <: AbstractArray{Float64,3}}
#     h = M.con
#     p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs
#     @cast A4[x, k, l, y] := A[x, (k, l), y] (k ∈ 1:maximum(p_rt))
#     @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lb))

#     R = zeros(size(A, 1), length(p_l), size(B, 1))
#     for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
#         AA = @view A4[:, p_rt[r], p_lt[l], :]
#         RR = @view RE[:, r, :]
#         BB = @view B4[:, p_lb[l], p_rb[r], :]
#         R[:, l, :] += h[p_l[l], p_r[r]] * (AA * RR * BB')
#     end
#     R
# end

# function update_env_right(
#     RE::S, A::S, M::T, B::S, ::Val{:c}
# ) where {T <: SparseVirtualTensor, S <: AbstractArray{Float64, 3}}
#     h = M.con
#     p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs
#     @cast A4[x, k, l, y] := A[x, (k, l), y] (k ∈ 1:maximum(p_rt))
#     @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lb))

#     R = zeros(size(A, 1), length(p_l), size(B, 1))
#     for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
#         AA = @view A4[:, p_lb[l], p_rb[r], :]
#         RR = @view RE[:, r, :]
#         BB = @view B4[:, p_rt[r], p_lt[l], :]
#         R[:, l, :] += h[p_l[l], p_r[r]] * (AA * RR * BB')
#     end
#     R
# end
