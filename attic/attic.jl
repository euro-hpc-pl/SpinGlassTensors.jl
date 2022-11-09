"""
$(TYPEDSIGNATURES)
"""
function _left_sweep_var_twosite!(env::Environment, Dcut::Int, tol::Number, args...)
    for site ∈ reverse(env.bra.sites[2:end])
        update_env_right!(env, site)
        A = project_ket_on_bra_twosite(env, site)
        @cast B[(x, y), (z, w)] := A[x, y, z, w]
        U, S, VV = svd(B, Dcut, tol, args...)
        V = VV'
        @cast C[x, σ, y] := V[x, (σ, y)] (σ ∈ 1:size(A, 3))
        env.bra[site] = C
        clear_env_containing_site!(env, site)
        if site == env.bra.sites[2]
            UU = U .* reshape(S, 1, :)
            @cast US[x, σ, y] := UU[(x, σ), y] (σ ∈ 1:size(A, 2))
            env.bra[env.bra.sites[1]] = US / norm(US)
            update_env_right!(env, env.bra.sites[2])
            update_env_right!(env, env.bra.sites[1])
        end
    end
end

#TODO: to be changed
function _right_sweep_var_twosite!(env::Environment, Dcut::Int, tol::Number, args...)
    for site ∈ env.bra.sites[1:end-1]
        site_r = _right_nbrs_site(site, env.bra.sites)
        update_env_left!(env, site)
        A = project_ket_on_bra_twosite(env, site_r)
        @cast B[(x, y), (z, w)] := A[x, y, z, w]
        U, S, V = svd(B, Dcut, tol, args...)
        @cast C[x, σ, y] := U[(x, σ), y] (σ ∈ 1:size(A, 2))
        env.bra[site] = C
        clear_env_containing_site!(env, site)
        if site_r == env.bra.sites[end]
            SV = S .* V'
            @cast SS[x, σ, y] := SV[x, (σ, y)] (σ ∈ 1:size(A, 3))
            env.bra[site_r] = SS ./ norm(SS)
            update_env_left!(env, site)
            update_env_left!(env, site_r)
        end
    end
end

# TODO: doesn't work
function compress_twosite!(
    bra::QMps, mpo::QMpo, ket::QMps, Dcut::Int, tol::Real=1E-8, max_sweeps::Int=4, args...
)
    env = Environment(bra, mpo, ket)
    overlap = Inf
    overlap_before = measure_env(env, last(env.bra.sites))
    for sweep ∈ 1:max_sweeps
        _left_sweep_var_twosite!(env, Dcut, tol, args...)
        _right_sweep_var_twosite!(env, Dcut, tol, args...)

        overlap = measure_env(env, last(env.bra.sites))

        Δ = abs(overlap_before - abs(overlap))
        @info "Convergence" Δ

        if Δ < tol
            @info "Finished in $sweep sweeps of $(max_sweeps)."
            return overlap
        else
            overlap_before = overlap
        end
    end
    overlap
end

# # """
# # $(TYPEDSIGNATURES)
# # """
# function project_ket_on_bra(
#     LE::S, B::S, M::T, RE::S, ::Val{:n}
# ) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}
#     h = M.con
#     if typeof(h) == SparseCentralTensor
#         h = dense_central_tensor(h)
#     end
#     p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

#     @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lb))

#     p_lb = projector_to_dense(p_lb)
#     p_l = projector_to_dense(p_l)
#     p_lt = projector_to_dense(p_lt)
#     @cast pl[bp, oc, tp, c] := p_lb[bp, c] * p_l[oc, c] * p_lt[tp, c]
#     @tensor LL[b, bp, oc, t, tp] := LE[b, c, t] * pl[bp, oc, tp, c]

#     p_rb = projector_to_dense(p_rb)
#     p_r = projector_to_dense(p_r)
#     p_rt = projector_to_dense(p_rt)
#     @cast pr[bp, oc, tp, c] := p_rb[bp, c] * p_r[oc, c] * p_rt[tp, c]
#     @tensor RR[t, tp, oc, b, bp] := RE[t, c, b] * pr[bp, oc, tp, c]

#     @tensor LR[tl, tlp, trp, tr] := LL[bl, blp, cl, tl, tlp] * RR[tr, trp, cr, br, brp] * B4[bl, blp, brp, br] * h[cl, cr] order = (cl, bl, blp, brp, br, cr)
#     @cast LR[l, (x, y), r] := LR[l, x, y, r]

#     LR ./ maximum(abs.(LR))
# end



# """
# $(TYPEDSIGNATURES)
# """
# function project_ket_on_bra(
#     LE::S, B::S, M::T, RE::S, ::Val{:c}
# ) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}
#     h = M.con
#     if typeof(h) == SparseCentralTensor
#         h = dense_central_tensor(h)
#     end
#     p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

#     @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lt))

#     pp_lb = projector_to_dense(p_lt)
#     pp_l = projector_to_dense(p_l)
#     pp_lt = projector_to_dense(p_lb)
#     @cast pl[bp, oc, tp, c] := pp_lb[bp, c] * pp_l[oc, c] * pp_lt[tp, c]
#     @tensor LL[b, bp, oc, t, tp] := LE[b, c, t] * pl[bp, oc, tp, c]

#     pp_rb = projector_to_dense(p_rt)
#     pp_r = projector_to_dense(p_r)
#     pp_rt = projector_to_dense(p_rb)
#     @cast pr[bp, oc, tp, c] := pp_rb[bp, c] * pp_r[oc, c] * pp_rt[tp, c]
#     @tensor RR[t, tp, oc, b, bp] := RE[t, c, b] * pr[bp, oc, tp, c]

#     @tensor LR[tl, tlp, trp, tr] := LL[bl, blp, cl, tl, tlp] * RR[tr, trp, cr, br, brp] * B4[bl, blp, brp, br] * h[cl, cr] order = (cl, bl, blp, brp, br, cr)
#     @cast LR[l, (x, y), r] := LR[l, x, y, r]

#     LR ./ maximum(abs.(LR))
# end

# """
# $(TYPEDSIGNATURES)
# """
# function project_ket_on_bra(
#     LE::S, B::S, M::T, RE::S, ::Val{:n}
# ) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}
#     h = M.con
#     if typeof(h) == SparseCentralTensor
#         h = cuda_dense_central_tensor(h)
#     else
#         h = CUDA.CuArray(h)
#     end
#     p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

#     A = CUDA.zeros(eltype(LE), maximum(p_lt) * maximum(p_rt), size(LE, 3) * size(RE, 1))

#     total_size = length(p_r)
#     batch_size = min(2^6, total_size)
#     from = 1
#     while from <= total_size
#         to = min(total_size, from + batch_size - 1)

#         @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lb))
#         B_d = permutedims(CUDA.CuArray(B4[:, p_lb, p_rb[from:to], :]), (1, 4, 2, 3))
#         @cast B_d[l, r, (s1, s2)] := B_d[l, r, s1, s2]

#         L_d = permutedims(CUDA.CuArray(LE), (3, 1, 2))
#         h_d = CUDA.CuArray(h[p_l, p_r[from:to]])
#         @cast Lh_d[l, r, (s1, s2)] := L_d[l, r, s1] * h_d[s1, s2]

#         R_d = permutedims(CUDA.CuArray(RE[:, from:to, :]), (3, 1, 2))
#         oo = CUDA.ones(eltype(R_d), length(p_l))
#         @cast R_d[l, r, (s1, s2)] := R_d[l, r, s2] * oo[s1]

#         LBR_d = Lh_d ⊠ B_d ⊠ R_d

#         p1, p2 = p_lt, p_rt[from:to]
#         pt = reshape(reshape(p1, :, 1) .+ maximum(p1) .* reshape(p2 .- 1, 1, :), :)
#         # pt = outer_projector(p_lt, p_rt[from:to])  cannot use it here

#         csrRowPtr = CuArray(collect(1:length(pt) + 1))
#         csrColInd = CuArray(pt)
#         csrNzVal = CUDA.ones(Float64, length(pt))
#         ipt = CUSPARSE.CuSparseMatrixCSC(csrRowPtr, csrColInd, csrNzVal, (maximum(p_lt) * maximum(p_rt), length(pt))) # transposed right here

#         @cast LBR_d[(l, r), s12] := LBR_d[l, r, s12]
#         A = A .+ ipt * LBR_d'

#         from = to + 1
#     end
#     @cast A[p12, l, r] := A[p12, (l, r)] (l ∈ 1:size(LE, 3))
#     Array(permutedims(A, (2, 1, 3)) ./ maximum(abs.(A)))
#     # @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lb))

#     # A = zeros(size(LE, 3), maximum(p_lt), maximum(p_rt), size(RE, 1))

#     # for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
#     #     le = @inbounds @view LE[:, l, :]
#     #     b = @inbounds @view B4[:, p_lb[l], p_rb[r], :]
#     #     re = @inbounds @view RE[:, r, :]
#     #     @inbounds A[:,  p_lt[l], p_rt[r], :] += h[p_l[l], p_r[r]] .* (le' * b * re')

#     # end
#     # @cast AA[l, (ũ, u), r] := A[l, ũ, u, r]
#     # AA
# end

# """
# $(TYPEDSIGNATURES)
# """
# function project_ket_on_bra(
#     LE::S, B::S, M::T, RE::S, ::Val{:c}
# ) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}
#     h = M.con
#     if typeof(h) == SparseCentralTensor
#         h = dense_central_tensor(h)
#     else
#         h = CUDA.CuArray(h)
#     end
#     p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

#     A = CUDA.zeros(eltype(LE), maximum(p_lb) * maximum(p_rb), size(LE, 3) * size(RE, 1))

#     total_size = length(p_r)
#     batch_size = min(2^6, total_size)
#     from = 1
#     while from <= total_size
#         to = min(total_size, from + batch_size - 1)

#         @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lt))
#         B_d = permutedims(CUDA.CuArray(B4[:, p_lt, p_rt[from:to], :]), (1, 4, 2, 3))
#         @cast B_d[l, r, (s1, s2)] := B_d[l, r, s1, s2]

#         L_d = permutedims(CUDA.CuArray(LE), (3, 1, 2))
#         h_d = CUDA.CuArray(h[p_l, p_r[from:to]])
#         @cast Lh_d[l, r, (s1, s2)] := L_d[l, r, s1] * h_d[s1, s2]

#         R_d = permutedims(CUDA.CuArray(RE[:, from:to, :]), (3, 1, 2))
#         oo = CUDA.ones(eltype(R_d), length(p_l))
#         @cast R_d[l, r, (s1, s2)] := R_d[l, r, s2] * oo[s1]

#         LBR_d = Lh_d ⊠ B_d ⊠ R_d

#         p1, p2 = p_lb, p_rb[from:to]
#         pb = reshape(reshape(p1, :, 1) .+ maximum(p1) .* reshape(p2 .- 1, 1, :), :)
#         # pb = outer_projector(p_lb, p_rb[from:to]) # cannot use here

#         csrRowPtr = CuArray(collect(1:length(pb) + 1))
#         csrColInd = CuArray(pb)
#         csrNzVal = CUDA.ones(Float64, length(pb))
#         ipb = CUSPARSE.CuSparseMatrixCSC(csrRowPtr, csrColInd, csrNzVal, (maximum(p_lb) * maximum(p_rb), length(pb))) # transposed right here

#         @cast LBR_d[(l, r), s12] := LBR_d[l, r, s12]
#         A = A .+ ipb * LBR_d'

#         from = to + 1
#     end
#     @cast A[p12, l, r] := A[p12, (l, r)] (l ∈ 1:size(LE, 3))
#     Array(permutedims(A, (2, 1, 3)) ./ maximum(abs.(A)))

#     # @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lt))

#     # A = zeros(size(LE, 3), maximum(p_lb), maximum(p_rb), size(RE, 1))
#     # for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
#     #     le = @inbounds @view LE[:, l, :]
#     #     b = @inbounds @view B4[:, p_lt[l], p_rt[r], :]
#     #     re = @inbounds @view RE[:, r, :]
#     #     @inbounds  A[:, p_lb[l], p_rb[r], :] += h[p_l[l], p_r[r]] .* (le' * b * re')
#     # end
#     # @cast AA[l, (ũ, u), r] := A[l, ũ, u, r]
#     # AA
# end

# """
# $(TYPEDSIGNATURES)
# """
# function project_ket_on_bra(
#     LE::S, B::S, M::T, RE::S, ::Val{:n}
# ) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}
#     h = M.con
#     if typeof(h) == SparseCentralTensor
#         h = dense_central_tensor(h)
#     end
#     p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

#     @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lb))

#     pls = projectors_to_sparse(p_lb, p_l, p_lt)
#     (a,b,c) = size(LE)
#     LE = permutedims(LE, (2, 1, 3))
#     @cast LEn[x, (y, z)] := LE[x, y, z]
#     LL = pls * LEn 
#     @cast LL[nbp, nc, ntp, nb, nt] := LL[(nbp, nc, ntp), (nb, nt)] (nbp ∈ 1:maximum(p_lb), nc ∈ 1:maximum(p_l), nb ∈ 1:a)
#     LL = permutedims(LL, (4, 1, 2, 5, 3))

#     prs = projectors_to_sparse(p_rb, p_r, p_rt)
#     (a,b,c) = size(RE)
#     RE = permutedims(RE, (2, 3, 1))
#     @cast REn[x, (y, z)] := RE[x, y, z]
#     RR = prs * REn 
#     @cast RR[nbp, nc, ntp, nb, nt] := RR[(nbp, nc, ntp), (nb, nt)] (nbp ∈ 1:maximum(p_rb), nc ∈ 1:maximum(p_r), nt ∈ 1:a)
#     RR = permutedims(RR, (5, 3, 2, 4, 1))

#     @tensor LR[tl, tlp, trp, tr] := LL[bl, blp, cl, tl, tlp] * RR[tr, trp, cr, br, brp] * B4[bl, blp, brp, br] * h[cl, cr] order = (cl, bl, blp, brp, br, cr)
#     @cast LR[l, (x, y), r] := LR[l, x, y, r]

#     LR ./ maximum(abs.(LR))
# end

# function project_ket_on_bra(
#     LE::S, B::S, M::T, RE::S, ::Val{:c}
# ) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}
#     h = M.con
#     if typeof(h) == SparseCentralTensor
#         h = dense_central_tensor(h)
#     end
#     p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

#     @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lt))

#     pls = projectors_to_sparse(p_lt, p_l, p_lb)
#     (a,b,c) = size(LE)
#     LE = permutedims(LE, (2, 1, 3))
#     @cast LEn[x, (y, z)] := LE[x, y, z]
#     LL = pls * LEn 
#     @cast LL[nbp, nc, ntp, nb, nt] := LL[(nbp, nc, ntp), (nb, nt)] (nbp ∈ 1:maximum(p_lt), nc ∈ 1:maximum(p_l), nb ∈ 1:a)
#     LL = permutedims(LL, (4, 1, 2, 5, 3))

#     prs = projectors_to_sparse(p_rt, p_r, p_rb)
#     (a,b,c) = size(RE)
#     RE = permutedims(RE, (2, 3, 1))
#     @cast REn[x, (y, z)] := RE[x, y, z]
#     RR = prs * REn 
#     @cast RR[nbp, nc, ntp, nb, nt] := RR[(nbp, nc, ntp), (nb, nt)] (nbp ∈ 1:maximum(p_rt), nc ∈ 1:maximum(p_r), nt ∈ 1:a)
#     RR = permutedims(RR, (5, 3, 2, 4, 1))

#     @tensor LR[tl, tlp, trp, tr] := LL[bl, blp, cl, tl, tlp] * RR[tr, trp, cr, br, brp] * B4[bl, blp, brp, br] * h[cl, cr] order = (cl, bl, blp, brp, br, cr)
#     @cast LR[l, (x, y), r] := LR[l, x, y, r]

#     LR ./ maximum(abs.(LR))
# end

# """
# $(TYPEDSIGNATURES)
# """
# function update_env_left(
#     LE::S, A::S, M::T, B::S, ::Val{:n}
# ) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}
#     h = M.con
#     if typeof(h) == SparseCentralTensor
#         h = dense_central_tensor(h)
#     end
#     p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

#     @cast A4[x, k, l, y] := A[x, (k, l), y] (k ∈ 1:maximum(p_lt))
#     @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lb))

#     ps = projectors_to_sparse(p_lb, p_l, p_lt)
#     (a,b,c) = size(LE)
#     LE = permutedims(LE, (2, 1, 3))
#     @cast LEn[x, (y, z)] := LE[x, y, z]
#     Ltemp = ps * LEn 

#     @cast Ltemp[nbp, nc, ntp, nb, nt] := Ltemp[(nbp, nc, ntp), (nb, nt)] (nbp ∈ 1:maximum(p_lb), nc ∈ 1:maximum(p_l), nb ∈ 1:a)
#     Ltemp = permutedims(Ltemp, (4, 1, 2, 5, 3))
#     @tensor Ltempnew[nb, nbp, nc, nt, ntp] := Ltemp[b, bp, oc, t, tp] * A4[t, tp, ntp, nt] * B4[b, bp, nbp, nb] * h[oc, nc]

#     a = size(Ltempnew, 1)
#     prs = projectors_to_sparse(p_rb, p_r, p_rt)
#     Ltempnew = permutedims(Ltempnew, (1, 4, 2, 3, 5))
#     @cast Ltempnew[(nb, nt), (nbp, nc, ntp)] :=  Ltempnew[nb, nt, nbp, nc, ntp]
#     Lnew = Ltempnew * prs #[(nb, nt), cc]
#     @cast Lnew[nb, nt, cc] := Lnew[(nb, nt), cc] (nb ∈ 1:a)
#     Lnew = permutedims(Lnew, (1, 3, 2))
#     #@tensor Lnew[nb, cc, nt] := Ltempnew[nb, nbp, nc, nt, ntp] * pr[nbp, nc, ntp, cc]

#     Lnew ./ maximum(abs.(Lnew))
# end

# """
# $(TYPEDSIGNATURES)
# """
# function update_env_left(
#     LE::S, A::S, M::T, B::S, ::Val{:n}
# ) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}
#     h = M.con
#     if typeof(h) == SparseCentralTensor
#         h = dense_central_tensor(h)
#     end
#     p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

#     #L = CUDA.zeros(eltype(LE), size(B, 3), size(A, 3), length(p_r))

#     @cast A4[x, k, l, y] := A[x, (k, l), y] (k ∈ 1:maximum(p_lt))
#     @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lb))

#     p_lb = projector_to_dense(p_lb)
#     p_l = projector_to_dense(p_l)
#     p_lt = projector_to_dense(p_lt)
#     @cast pl[bp, oc, tp, c] := p_lb[bp, c] * p_l[oc, c] * p_lt[tp, c] 
#     @tensor LL[b, bp, oc, t, tp] := LE[b, c, t] * pl[bp, oc, tp, c]

#     #  ps = projectors_to_sparse(p_lb, p_l, p_lt) -> sparse[oc, nc]
#     #  Ltemp = ps[nc, c] * LE[b, c, t]
#     #  @cast Ltemp[nb, nbp, nc, nt, ntp] := Ltemp[nb, (nbp, nc, ntp), nt] (nbp ∈ 1:maximum(p_lb), nc ∈ 1:maximum(p_l))

#     @tensor Ltemp[nb, nbp, nc, nt, ntp] := LL[b, bp, oc, t, tp] * A4[t, tp, ntp, nt] * B4[b, bp, nbp, nb] * h[oc, nc]

#     p_rb = projector_to_dense(p_rb)
#     p_r = projector_to_dense(p_r)
#     p_rt = projector_to_dense(p_rt)
#     @cast pr[bp, oc, tp, c] := p_rb[bp, c] * p_r[oc, c] * p_rt[tp, c]
#     @tensor Lnew[nb, cc, nt] := Ltemp[nb, nbp, nc, nt, ntp] * pr[nbp, nc, ntp, cc]

#     Lnew ./ maximum(abs.(Lnew))
# end


# """
# $(TYPEDSIGNATURES)
# """
# function update_env_left(
#     LE::S, A::S, M::T, B::S, ::Val{:n}
# ) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}
#     h = M.con
#     if typeof(h) == SparseCentralTensor
#         h = cuda_dense_central_tensor(h)
#     else
#         h = CUDA.CuArray(h)
#     end
#     p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

#     L = CUDA.zeros(eltype(LE), size(B, 3), size(A, 3), length(p_r))

#     total_size = length(p_r)
#     batch_size = min(2^6, total_size)
#     from = 1
#     while from <= total_size
#         to = min(total_size, from + batch_size - 1)

#         @cast A4[x, k, l, y] := A[x, (k, l), y] (k ∈ 1:maximum(p_lt))
#         A_d = permutedims(CUDA.CuArray(A4[:, p_lt, p_rt[from:to], :]), (1, 4, 2, 3))
#         @cast A_d[l, r, (s1, s2)] := A_d[l, r, s1, s2]

#         @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lb))
#         B_d = permutedims(CUDA.CuArray(B4[:, p_lb, p_rb[from:to], :]), (4, 1, 2, 3))
#         @cast B_d[l, r, (s1, s2)] := B_d[l, r, s1, s2]

#         h_d = CUDA.CuArray(h[p_l, p_r[from:to]])
#         L_d = permutedims(CUDA.CuArray(LE), (1, 3, 2))
#         @cast Lh_d[l, r, (s1, s2)] := L_d[l, r, s1] * h_d[s1, s2]

#         LhAB_d = B_d ⊠ Lh_d ⊠ A_d

#         @cast LhAB_d[l, r, s1, s2] := LhAB_d[l, r, (s1, s2)] (s2 ∈ 1:(to - from + 1))
#         L[:, :, from:to] = dropdims(sum(LhAB_d, dims=3), dims=3)
#         from = to + 1
#     end
#     Array(permutedims(L, (1, 3, 2)) ./ maximum(abs.(L)))
# end


# """
# $(TYPEDSIGNATURES)
# """
# function update_env_left(
#     LE::S, A::S, M::T, B::S, ::Val{:c}
# ) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}
#     h = M.con
#     if typeof(h) == SparseCentralTensor
#         h = dense_central_tensor(h)
#     end
#     p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

#     @cast A4[x, k, l, y] := A[x, (k, l), y] (k ∈ 1:maximum(p_lb))
#     @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lt))

#     ps = projectors_to_sparse(p_lb, p_l, p_lt)
#     (a,b,c) = size(LE)
#     LE = permutedims(LE, (2, 1, 3))
#     @cast LEn[x, (y, z)] := LE[x, y, z]
#     Ltemp = ps * LEn 

#     @cast Ltemp[nbp, nc, ntp, nb, nt] := Ltemp[(nbp, nc, ntp), (nb, nt)] (nbp ∈ 1:maximum(p_lb), nc ∈ 1:maximum(p_l), nb ∈ 1:a)
#     Ltemp = permutedims(Ltemp, (4, 1, 2, 5, 3))
#     @tensor Ltempnew[nb, ntp, nc, nt, nbp] := Ltemp[b, bp, oc, t, tp] * A4[t, bp, ntp, nt] * B4[b, tp, nbp, nb] * h[oc, nc]
#     #Ltempnew[nb, nbp, nc, nt, ntp] := Ltemp[b, bp, oc, t, tp] * A4[t, tp, ntp, nt] * B4[b, bp, nbp, nb] * h[oc, nc]

#     a = size(Ltempnew, 1)
#     prs = projectors_to_sparse(p_rb, p_r, p_rt)
#     Ltempnew = permutedims(Ltempnew, (1, 4, 5, 3, 2))
#     @cast Ltempnew[(nb, nt), (nbp, nc, ntp)] :=  Ltempnew[nb, nt, nbp, nc, ntp]
#     Lnew = Ltempnew * prs #[(nb, nt), cc]
#     @cast Lnew[nb, nt, cc] := Lnew[(nb, nt), cc] (nb ∈ 1:a)
#     Lnew = permutedims(Lnew, (1, 3, 2))
#     #@tensor Lnew[nb, cc, nt] := Ltempnew[nb, nbp, nc, nt, ntp] * pr[nbp, nc, ntp, cc]

#     Lnew ./ maximum(abs.(Lnew))
# end


# """
# $(TYPEDSIGNATURES)
# """
# function update_env_left(
#     LE::S, A::S, M::T, B::S, ::Val{:c}
# ) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}
#     h = M.con
#     if typeof(h) == SparseCentralTensor
#         h = dense_central_tensor(h)
#     end
#     p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

#     #L = CUDA.zeros(eltype(LE), size(B, 3), size(A, 3), length(p_r))

#     @cast A4[x, k, l, y] := A[x, (k, l), y] (k ∈ 1:maximum(p_lb))
#     @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lt))

#     p_lb = projector_to_dense(p_lb)
#     p_l = projector_to_dense(p_l)
#     p_lt = projector_to_dense(p_lt)
#     @cast pl[bp, oc, tp, c] := p_lb[bp, c] * p_l[oc, c] * p_lt[tp, c]
#     @tensor LL[b, bp, oc, t, tp] := LE[b, c, t] * pl[bp, oc, tp, c]

#     #  ps = projectors_to_sparse(p_lb, p_l, p_lt) -> sparse[oc, nc]
#     #  Ltemp = ps[nc, c] * LE[b, c, t]
#     #  @cast Ltemp[nb, nbp, nc, nt, ntp] := A[nb, (nbp, nc, ntp), nt] (nbp ∈ 1:maximum(p_lb), nc ∈ 1:maximum(p_l))

#     @tensor Ltemp[nb, ntp, nc, nt, nbp] := LL[b, bp, oc, t, tp] * A4[t, bp, ntp, nt] * B4[b, tp, nbp, nb] * h[oc, nc]

#     p_rb = projector_to_dense(p_rb)
#     p_r = projector_to_dense(p_r)
#     p_rt = projector_to_dense(p_rt)
#     @cast pr[bp, oc, tp, c] := p_rb[bp, c] * p_r[oc, c] * p_rt[tp, c]
#     @tensor Lnew[nb, cc, nt] := Ltemp[nb, ntp, nc, nt, nbp] * pr[ntp, nc, nbp, cc]

#     Lnew ./ maximum(abs.(Lnew))
# end

# """
# $(TYPEDSIGNATURES)
# """
# function update_env_left(
#     LE::S, A::S, M::T, B::S, ::Val{:c}
# ) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}
#     h = M.con
#     if typeof(h) == SparseCentralTensor
#         h = cuda_dense_central_tensor(h)
#     else
#         h = CUDA.CuArray(h)
#     end
#     p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

#     L = CUDA.zeros(eltype(LE), size(B, 3), size(A, 3), length(p_r))

#     total_size = length(p_r)
#     batch_size = min(2^6, total_size)
#     from = 1
#     while from <= total_size
#         to = min(total_size, from + batch_size - 1)

#         @cast A4[x, k, l, y] := A[x, (k, l), y] (k ∈ 1:maximum(p_lb))
#         A_d = permutedims(CUDA.CuArray(A4[:, p_lb, p_rb[from:to], :]), (1, 4, 2, 3))
#         @cast A_d[l, r, (s1, s2)] := A_d[l, r, s1, s2]

#         @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lt))
#         B_d = permutedims(CUDA.CuArray(B4[:, p_lt, p_rt[from:to], :]), (4, 1, 2, 3))
#         @cast B_d[l, r, (s1, s2)] := B_d[l, r, s1, s2]

#         h_d = CUDA.CuArray(h[p_l, p_r[from:to]])
#         L_d = permutedims(CUDA.CuArray(LE), (1, 3, 2))
#         @cast Lh_d[l, r, (s1, s2)] := L_d[l, r, s1] * h_d[s1, s2]

#         LhAB_d = B_d ⊠ Lh_d ⊠ A_d

#         @cast LhAB_d[l, r, s1, s2] := LhAB_d[l, r, (s1, s2)] (s2 ∈ 1:(to - from + 1))
#         L[:, :, from:to] = dropdims(sum(LhAB_d, dims=3), dims=3)
#         from = to + 1
#     end
#     Array(permutedims(L, (1, 3, 2)) ./ maximum(abs.(L)))
# end


# """
# $(TYPEDSIGNATURES)
# """
# function update_env_right(
#     RE::S, A::S, M::T, B::S, ::Val{:n}
# ) where {T <: SparseVirtualTensor, S <: AbstractArray{Float64,3}}
#     h = M.con
#     if typeof(h) == SparseCentralTensor
#         h = dense_central_tensor(h)
#     end
#     p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

#     @cast A4[x, k, l, y] := A[x, (k, l), y] (k ∈ 1:maximum(p_lt))
#     @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lb))

#     ps = projectors_to_sparse(p_rb, p_r, p_rt)
#     (a,b,c) = size(RE)
#     RE = permutedims(RE, (2, 3, 1))
#     @cast REn[x, (y, z)] := RE[x, y, z]
#     Rtemp = ps * REn 

#     @cast Rtemp[nbp, nc, ntp, nb, nt] := Rtemp[(nbp, nc, ntp), (nb, nt)] (nbp ∈ 1:maximum(p_rb), nc ∈ 1:maximum(p_r), nt ∈ 1:a)
#     Rtemp = permutedims(Rtemp, (5, 3, 2, 4, 1))
#     @tensor Rtempnew[nt, ntp, nc, nb, nbp] := Rtemp[t, tp, oc, b, bp] * A4[nt, ntp, tp, t] * B4[nb, nbp, bp, b] * h[nc, oc]
#     #Ltempnew[nb, nbp, nc, nt, ntp] := Ltemp[b, bp, oc, t, tp] * A4[t, tp, ntp, nt] * B4[b, bp, nbp, nb] * h[oc, nc]

#     a = size(Rtempnew, 1)
#     pls = projectors_to_sparse(p_lb, p_l, p_lt)
#     Rtempnew = permutedims(Rtempnew, (1, 4, 5, 3, 2))
#     @cast Rtempnew[(nt, nb), (ntp, nc, nbp)] :=  Rtempnew[nt, nb, ntp, nc, nbp]
#     Rnew = Rtempnew * pls #[(nb, nt), cc]
#     @cast Rnew[nt, nb, cc] := Rnew[(nt, nb), cc] (nt ∈ 1:a)
#     Rnew = permutedims(Rnew, (1, 3, 2))
   
#     Rnew ./ maximum(abs.(Rnew))
# end

# """
# $(TYPEDSIGNATURES)
# """
# function update_env_right(
#     RE::S, A::S, M::T, B::S, ::Val{:n}
# ) where {T <: SparseVirtualTensor, S <: AbstractArray{Float64,3}}
#     h = M.con
#     if typeof(h) == SparseCentralTensor
#         h = dense_central_tensor(h)
#     end
#     p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

#     #L = CUDA.zeros(eltype(LE), size(B, 3), size(A, 3), length(p_r))

#     @cast A4[x, k, l, y] := A[x, (k, l), y] (k ∈ 1:maximum(p_lt))
#     @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lb))

#     p_rb = projector_to_dense(p_rb)
#     p_r = projector_to_dense(p_r)
#     p_rt = projector_to_dense(p_rt)
#     @cast pr[bp, oc, tp, c] := p_rb[bp, c] * p_r[oc, c] * p_rt[tp, c]

#     @tensor RR[t, tp, oc, b, bp] := RE[t, c, b] * pr[bp, oc, tp, c]

#     #  ps = projectors_to_sparse(p_lb, p_l, p_lt) -> sparse[oc, nc]
#     #  Ltemp = ps[nc, c] * LE[b, c, t]
#     #  @cast Ltemp[nb, nbp, nc, nt, ntp] := A[nb, (nbp, nc, ntp), nt] (nbp ∈ 1:maximum(p_lb), nc ∈ 1:maximum(p_l))

#     @tensor Rtemp[nt, ntp, nc, nb, nbp] := RR[t, tp, oc, b, bp] * A4[nt, ntp, tp, t] * B4[nb, nbp, bp, b] * h[nc, oc]

#     p_lb = projector_to_dense(p_lb)
#     p_l = projector_to_dense(p_l)
#     p_lt = projector_to_dense(p_lt)
#     @cast pl[bp, oc, tp, c] := p_lb[bp, c] * p_l[oc, c] * p_lt[tp, c]

#     @tensor Rnew[nt, cc, nb] := Rtemp[nt, ntp, nc, nb, nbp] * pl[nbp, nc, ntp, cc]

#     Rnew ./ maximum(abs.(Rnew))
# end

# """
# $(TYPEDSIGNATURES)
# """
# function update_env_right(
#     RE::S, A::S, M::T, B::S, ::Val{:n}
# ) where {T <: SparseVirtualTensor, S <: AbstractArray{Float64,3}}
#     h = M.con
#     if typeof(h) == SparseCentralTensor
#         h = cuda_dense_central_tensor(h)
#     else
#         h = CUDA.CuArray(h)
#     end
#     p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

#     total_size = length(p_l)
#     R = CUDA.zeros(eltype(RE), size(A, 1), size(B, 1), total_size)
#     batch_size = min(2^6, total_size)
#     from = 1
#     while from <= total_size
#         to = min(total_size, from + batch_size - 1)

#         @cast A4[x, k, l, y] := A[x, (k, l), y] (k ∈ 1:maximum(p_lt))
#         A_d = permutedims(CUDA.CuArray(A4[:, p_lt[from:to], p_rt, :]), (1, 4, 2, 3))
#         @cast A_d[l, r, (s1, s2)] := A_d[l, r, s1, s2]

#         @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lb))
#         B_d = permutedims(CUDA.CuArray(B4[:, p_lb[from:to], p_rb, :]), (4, 1, 2, 3))
#         @cast B_d[l, r, (s1, s2)] := B_d[l, r, s1, s2]

#         h_d = CUDA.CuArray(h[p_l[from:to], p_r])
#         R_d = permutedims(CUDA.CuArray(RE), (1, 3, 2))
#         @cast Rh_d[l, r, (s1, s2)] := R_d[l, r, s2] * h_d[s1, s2]

#         RhAB_d = A_d ⊠ Rh_d ⊠ B_d

#         @cast RhAB_d[l, r, s1, s2] := RhAB_d[l, r, (s1, s2)] (s1 ∈ 1:(to - from + 1))
#         R[:, :, from:to] = dropdims(sum(RhAB_d, dims=4), dims=4)
#         from = to + 1
#     end
#     Array(permutedims(R, (1, 3, 2)) ./ maximum(abs.(R)))

#     # R = zeros(size(A, 1), length(p_l), size(B, 1))
#     # for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
#     #     AA = @inbounds @view A4[:, p_lt[l], p_rt[r], :]
#     #     RR = @inbounds @view RE[:, r, :]
#     #     BB = @inbounds @view B4[:, p_lb[l], p_rb[r], :]
#     #     @inbounds R[:, l, :] += h[p_l[l], p_r[r]] * (AA * RR * BB')
#     # end
#     # R
# end


# """
# $(TYPEDSIGNATURES)
# """
# function update_env_right(
#     RE::S, A::S, M::T, B::S, ::Val{:c}
# ) where {T <: SparseVirtualTensor, S <: AbstractArray{Float64,3}}
# h = M.con
# if typeof(h) == SparseCentralTensor
#     h = dense_central_tensor(h)
# end
# p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

# @cast A4[x, k, l, y] := A[x, (k, l), y] (k ∈ 1:maximum(p_lb))
# @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lt))

# ps = projectors_to_sparse(p_rb, p_r, p_rt)
# (a,b,c) = size(RE)
# RE = permutedims(RE, (2, 3, 1))
# @cast REn[x, (y, z)] := RE[x, y, z]
# Rtemp = ps * REn 

# @cast Rtemp[nbp, nc, ntp, nb, nt] := Rtemp[(nbp, nc, ntp), (nb, nt)] (nbp ∈ 1:maximum(p_rb), nc ∈ 1:maximum(p_r), nt ∈ 1:a)
# Rtemp = permutedims(Rtemp, (5, 3, 2, 4, 1))
# @tensor Rtempnew[nt, nbp, nc, nb, ntp] := Rtemp[t, tp, oc, b, bp] * A4[nt, ntp, bp, t] * B4[nb, nbp, tp, b] * h[nc, oc]

# a = size(Rtempnew, 1)
# pls = projectors_to_sparse(p_lb, p_l, p_lt)
# Rtempnew = permutedims(Rtempnew, (1, 4, 2, 3, 5))
# @cast Rtempnew[(nt, nb), (ntp, nc, nbp)] :=  Rtempnew[nt, nb, ntp, nc, nbp]
# Rnew = Rtempnew * pls #[(nb, nt), cc]
# @cast Rnew[nt, nb, cc] := Rnew[(nt, nb), cc] (nt ∈ 1:a)
# Rnew = permutedims(Rnew, (1, 3, 2))

# Rnew ./ maximum(abs.(Rnew))
# end

# """
# $(TYPEDSIGNATURES)
# """
# function update_env_right(
#     RE::S, A::S, M::T, B::S, ::Val{:c}
# ) where {T <: SparseVirtualTensor, S <: AbstractArray{Float64,3}}
#     h = M.con
#     if typeof(h) == SparseCentralTensor
#         h = dense_central_tensor(h)
#     end
#     p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

#     #L = CUDA.zeros(eltype(LE), size(B, 3), size(A, 3), length(p_r))

#     @cast A4[x, k, l, y] := A[x, (k, l), y] (k ∈ 1:maximum(p_lb))
#     @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lt))

#     p_rb = projector_to_dense(p_rb)
#     p_r = projector_to_dense(p_r)
#     p_rt = projector_to_dense(p_rt)
#     @cast pr[bp, oc, tp, c] := p_rb[bp, c] * p_r[oc, c] * p_rt[tp, c]

#     @tensor RR[t, tp, oc, b, bp] := RE[t, c, b] * pr[bp, oc, tp, c]

#     #  ps = projectors_to_sparse(p_lb, p_l, p_lt) -> sparse[oc, nc]
#     #  Ltemp = ps[nc, c] * LE[b, c, t]
#     #  @cast Ltemp[nb, nbp, nc, nt, ntp] := A[nb, (nbp, nc, ntp), nt] (nbp ∈ 1:maximum(p_lb), nc ∈ 1:maximum(p_l))

#     @tensor Rtemp[nt, nbp, nc, nb, ntp] := RR[t, tp, oc, b, bp] * A4[nt, ntp, bp, t] * B4[nb, nbp, tp, b] * h[nc, oc]

#     p_lb = projector_to_dense(p_lb)
#     p_l = projector_to_dense(p_l)
#     p_lt = projector_to_dense(p_lt)
#     @cast pl[bp, oc, tp, c] := p_lb[bp, c] * p_l[oc, c] * p_lt[tp, c]

#     @tensor Rnew[nt, cc, nb] := Rtemp[nt, nbp, nc, nb, ntp] * pl[ntp, nc, nbp, cc] #c

#     Rnew ./ maximum(abs.(Rnew))
# end


# """
# $(TYPEDSIGNATURES)
# """
# function update_env_right(
#     RE::S, A::S, M::T, B::S, ::Val{:c}
# ) where {T <: SparseVirtualTensor, S <: AbstractArray{Float64, 3}}
#     h = M.con
#     if typeof(h) == SparseCentralTensor
#         h = cuda_dense_central_tensor(h)
#     else
#         h = CUDA.CuArray(h)
#     end
#     p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

#     total_size = length(p_l)
#     R = CUDA.zeros(eltype(RE), size(A, 1), size(B, 1), total_size)
#     batch_size = min(2^6, total_size)
#     from = 1
#     while from <= total_size
#         to = min(total_size, from + batch_size - 1)

#         @cast A4[x, k, l, y] := A[x, (k, l), y] (k ∈ 1:maximum(p_lb))
#         A_d = permutedims(CUDA.CuArray(A4[:, p_lb[from:to], p_rb, :]), (1, 4, 2, 3))
#         @cast A_d[l, r, (s1, s2)] := A_d[l, r, s1, s2]

#         @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lt))
#         B_d = permutedims(CUDA.CuArray(B4[:, p_lt[from:to], p_rt, :]), (4, 1, 2, 3))
#         @cast B_d[l, r, (s1, s2)] := B_d[l, r, s1, s2]

#         h_d = CUDA.CuArray(h[p_l[from:to], p_r])
#         R_d = permutedims(CUDA.CuArray(RE), (1, 3, 2))
#         @cast Rh_d[l, r, (s1, s2)] := R_d[l, r, s2] * h_d[s1, s2]

#         RhAB_d = A_d ⊠ Rh_d ⊠ B_d

#         @cast RhAB_d[l, r, s1, s2] := RhAB_d[l, r, (s1, s2)] (s1 ∈ 1:(to - from + 1))
#         R[:, :, from:to] = dropdims(sum(RhAB_d, dims=4), dims=4)
#         from = to + 1
#     end
#     Array(permutedims(R, (1, 3, 2)) ./ maximum(abs.(R)))


#     # @cast A4[x, k, l, y] := A[x, (k, l), y] (k ∈ 1:maximum(p_lb))
#     # @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lt))

#     # R = zeros(size(A, 1), length(p_l), size(B, 1))
#     # for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
#     #     AA = @inbounds @view A4[:, p_lb[l], p_rb[r], :]
#     #     RR = @inbounds @view RE[:, r, :]
#     #     BB = @inbounds @view B4[:, p_lt[l], p_rt[r], :]
#     #     @inbounds R[:, l, :] += h[p_l[l], p_r[r]] * (AA * RR * BB')
#     # end
#     # R
# end


"""
$(TYPEDSIGNATURES)
"""
function update_env_left(
    L::S, A::S, M::T, B::S, ::Val{:n}
) where {S <: Array{Real, 3}, T <: SparsePegasusSquareTensor}
    pr, pd = M.projs
    p1l, p2l, p1u, p2u = M.bnd_projs

    ipr = cuIdentity(eltype(L), maximum(pr))[pr, :]
    lel1, lel2, leu1, leu2 = CUDA.CuArray.(M.bnd_exp)

    A_d = permutedims(CUDA.CuArray(A), (1, 3, 2))
    L_d = permutedims(CUDA.CuArray(L), (1, 3, 2))
    B_d = permutedims(CUDA.CuArray(B), (3, 1, 2))[:, :, pd]

    @cast ll[x, y, z, l1] := L_d[x, y, z] * lel1[z, l1]
    @tensor LL[x, y, l1, l2] := ll[x, y, z, l1] * lel2[z, l2] # D x D x 2^12 x 2^6

    @cast lu[x, y, z, u1] := A_d[x, y, z] * leu1[z, u1]
    @tensor AA[x, y, u1, u2] := lu[x, y, z, u1] * leu2[z, u2]

    LL = LL[:, :, p1l, p2l]
    @cast LL[x, y, (s, r)] := LL[x, y, s, r]

    AA = AA[:, :, p1u, p2u]
    @cast AA[x, y, (s, r)] := AA[x, y, s, r]

    BB = repeat(B_d, outer=(1, 1, 1, size(pr, 1)))
    @cast BB[x, y, (z, σ)] := BB[x, y, z, σ] (σ ∈ 1:size(pr, 1))

    Lnew_no_le = BB ⊠ LL ⊠ AA  # broadcast over dims = 3

    Ln = Lnew_no_le .* reshape(CUDA.CuArray(M.loc_exp), 1, 1, :)
    @reduce Lnew[x, y, σ] := sum(z) Ln[x, y, (z, σ)] (σ ∈ 1:size(pr, 1))

    @tensor ret[x, y, r] := Lnew[x, y, z] * ipr[z, r]

    Array(permutedims(ret, (1, 3, 2)) ./ maximum(abs.(ret)))
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_left(
    L::S, A::S, M::T, B::S, ::Val{:c}
) where {S <: Array{Real, 3}, T <: SparsePegasusSquareTensor}
    pr, pd = M.projs
    p1l, p2l, p1u, p2u = M.bnd_projs

    lel1, lel2, leu1, leu2 = CUDA.CuArray.(M.bnd_exp)
    ipr = cuIdentity(eltype(L), maximum(pr))[pr, :]

    A_d = permutedims(CUDA.CuArray(A), (1, 3, 2))[:, :, pd]
    L_d = permutedims(CUDA.CuArray(L), (1, 3, 2))
    B_d = permutedims(CUDA.CuArray(B), (3, 1, 2))

    @cast ll[x, y, z, l1] := L_d[x, y, z] * lel1[z, l1]
    @tensor LL[x, y, l1, l2] := ll[x, y, z, l1] * lel2[z, l2] # D x D x 2^12 x 2^6

    @cast lu[x, y, z, u1] := B_d[x, y, z] * leu1[z, u1]
    @tensor BB[x, y, u1, u2] := lu[x, y, z, u1] * leu2[z, u2]

    LL = LL[:, :, p1l, p2l]
    @cast LL[x, y, (s, r)] := LL[x, y, s, r]

    BB = BB[:, :, p1u, p2u]
    @cast BB[x, y, (s, r)] := BB[x, y, s, r]

    AA = repeat(A_d, outer=(1, 1, 1, size(pr, 1)))
    @cast AA[x, y, (z, σ)] := AA[x, y, z, σ] (σ ∈ 1:size(pr, 1))

    Lnew_no_le = BB ⊠ LL ⊠ AA  # broadcast over dims = 3

    Ln = Lnew_no_le .* reshape(CUDA.CuArray(M.loc_exp), 1, 1, :)
    @reduce Lnew[x, y, σ] := sum(z) Ln[x, y, (z, σ)] (σ ∈ 1:size(pr, 1))

    @tensor ret[x, y, r] := Lnew[x, y, z] * ipr[z, r]

    Array(permutedims(ret, (1, 3, 2)) ./ maximum(abs.(ret)))
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_right(
    R::S, A::S, M::T, B::S, ::Val{:n}
) where {T <: SparsePegasusSquareTensor, S <: Array{Real, 3}}
    pr, pd = M.projs
    p1l, p2l, p1u, p2u = M.bnd_projs
    lel1, lel2, leu1, leu2 = CUDA.CuArray.(M.bnd_exp)

    ip1l = cuIdentity(eltype(R), maximum(p1l))[p1l, :]
    ip2l = cuIdentity(eltype(R), maximum(p2l))[p2l, :]

    A_d = permutedims(CUDA.CuArray(A), (1, 3, 2))
    R_d = permutedims(CUDA.CuArray(R), (1, 3, 2))[:, :, pr]
    B_d = permutedims(CUDA.CuArray(B), (3, 1, 2))[:, :, pd]

    @cast R_d[x, y, _, z] := R_d[x, y, z]

    @cast lu[x, y, z, l1] := A_d[x, y, z] * leu1[z, l1]
    @tensor AA[x, y, l1, l2] := lu[x, y, z, l1] * leu2[z, l2] # D x D x 2^12 x 2^6

    AA = AA[:, :, p1u, p2u]
    @cast AA[x, y, (s, r)] := AA[x, y, s, r]

    RR = repeat(R_d, outer=(1, 1, size(pd, 1), 1))
    @cast RR[x, y, (z, σ)] := RR[x, y, z, σ] (σ ∈ 1:size(pd, 1))

    BB = repeat(B_d, outer=(1, 1, 1, size(pr, 1)))
    @cast BB[x, y, (z, σ)] := BB[x, y, z, σ] (σ ∈ 1:size(pr, 1))

    Rnew_no_le = AA ⊠ RR ⊠ BB  # broadcast over dims = 3

    Rn = Rnew_no_le .* reshape(CUDA.CuArray(M.loc_exp), 1, 1, :)
    @cast Rnew[x, y, η, σ] := Rn[x, y, (η, σ)] (σ ∈ 1:size(pr, 1))

    @cast ll[x, y, z] := lel1[x, y] * lel2[x, z]
    @tensor ret[x, y, l] := Rnew[x, y, s1, s2] * ip1l[s1, l1] * ip2l[s2, l2] * ll[l, l1, l2]  order=(s2, s1, l1, l2)

    Array(permutedims(ret, (1, 3, 2)) ./ maximum(abs.(ret)))
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_right(
    R::S, A::S, M::T, B::S, ::Val{:c}
) where {T <: SparsePegasusSquareTensor, S <: Array{Real, 3}}
    pr, pd = M.projs
    p1l, p2l, p1u, p2u = M.bnd_projs
    lel1, lel2, leu1, leu2 = CUDA.CuArray.(M.bnd_exp)

    ip1l = cuIdentity(eltype(R), maximum(p1l))[p1l, :]
    ip2l = cuIdentity(eltype(R), maximum(p2l))[p2l, :]

    A_d = permutedims(CUDA.CuArray(A), (1, 3, 2))[:, :, pd]
    R_d = permutedims(CUDA.CuArray(R), (1, 3, 2))[:, :, pr]
    B_d = permutedims(CUDA.CuArray(B), (3, 1, 2))

    @cast R_d[x, y, _, z] := R_d[x, y, z]

    @cast lu[x, y, z, u1] := B_d[x, y, z] * leu1[z, u1]
    @tensor BB[x, y, u1, u2] := lu[x, y, z, u1] * leu2[z, u2] # D x D x 2^12 x 2^6

    BB = BB[:, :, p1u, p2u]
    @cast BB[x, y, (s, r)] := BB[x, y, s, r]

    RR = repeat(R_d, outer=(1, 1, size(pd, 1), 1))
    @cast RR[x, y, (z, σ)] := RR[x, y, z, σ] (σ ∈ 1:size(pd, 1))

    AA = repeat(A_d, outer=(1, 1, 1, size(pr, 1)))
    @cast AA[x, y, (z, σ)] := AA[x, y, z, σ] (σ ∈ 1:size(pr, 1))

    Rnew_no_le = AA ⊠ RR ⊠ BB  # broadcast over dims = 3

    Rn = Rnew_no_le .* reshape(CUDA.CuArray(M.loc_exp), 1, 1, :)
    @cast Rnew[x, y, η, σ] := Rn[x, y, (η, σ)] (σ ∈ 1:size(pr, 1))

    @cast ll[x, y, z] := lel1[x, y] * lel2[x, z]
    @tensor ret[x, y, l] := Rnew[x, y, s1, s2] * ip1l[s1, l1] * ip2l[s2, l2] *  ll[l, l1, l2]  order=(s2, s1, l1, l2)

    Array(permutedims(ret, (1, 3, 2)) ./ maximum(abs.(ret)))
end

function project_ket_on_bra(
    L::S, B::S, M::T, R::S, ::Val{:n}
) where {S <: Array{Real, 3}, T <: SparsePegasusSquareTensor}
    pr, pd = M.projs
    p1l, p2l, p1u, p2u = M.bnd_projs
    lel1, lel2, leu1, leu2 = CUDA.CuArray.(M.bnd_exp)

    ip1u = cuIdentity(eltype(L), maximum(p1u))[p1u, :]
    ip2u = cuIdentity(eltype(L), maximum(p2u))[p2u, :]

    L_d = permutedims(CUDA.CuArray(L), (3, 1, 2))
    B_d = permutedims(CUDA.CuArray(B), (1, 3, 2))[:, :, pd]
    R_d = permutedims(CUDA.CuArray(R), (3, 1, 2))[:, :, pr]

    @cast R_d[x, y, _, z] := R_d[x, y, z]

    @cast ll[x, y, z, l1] := L_d[x, y, z] * lel1[z, l1]
    @tensor LL[x, y, l1, l2] := ll[x, y, z, l1] * lel2[z, l2] # D x D x 2^12 x 2^6

    LL = LL[:, :, p1l, p2l]
    @cast LL[x, y, (s, r)] := LL[x, y, s, r]

    RR = repeat(R_d, outer=(1, 1, size(pd, 1), 1))
    @cast RR[x, y, (z, σ)] := RR[x, y, z, σ] (σ ∈ 1:size(pd, 1))

    BB = repeat(B_d, outer=(1, 1, 1, size(pr, 1)))
    @cast BB[x, y, (z, σ)] := BB[x, y, z, σ] (σ ∈ 1:size(pr, 1))

    Anew_no_le = LL ⊠ BB ⊠ RR

    An = Anew_no_le .* reshape(CUDA.CuArray(M.loc_exp), 1, 1, :)
    @cast Anew[x, y, η, σ] := An[x, y, (η, σ)] (σ ∈ 1:size(pr, 1))

    @cast lu[x, y, z] := leu1[x, y] * leu2[x, z]
    @tensor ret[x, y, u] := Anew[x, y, s1, s2] * ip1u[s1, u1] * ip2u[s2, u2] *  lu[u, u1, u2]  order=(s2, s1, u1, u2)

    Array(permutedims(ret, (1, 3, 2)) ./ maximum(abs.(ret)))
end

"""
$(TYPEDSIGNATURES)
"""
function project_ket_on_bra(
    L::S, B::S, M::T, R::S, ::Val{:c}
) where {S <: Array{Real, 3}, T <: SparsePegasusSquareTensor}
    pr, pd = M.projs
    p1l, p2l, p1u, p2u = M.bnd_projs
    lel1, lel2, leu1, leu2 = CUDA.CuArray.(M.bnd_exp)

    ipd = cuIdentity(eltype(L), maximum(pd))[pd, :]

    L_d = permutedims(CUDA.CuArray(L), (3, 1, 2))
    B_d = permutedims(CUDA.CuArray(B), (1, 3, 2))
    R_d = permutedims(CUDA.CuArray(R), (3, 1, 2))[:, :, pr]

    @cast R_d[x, y, _, z] := R_d[x, y, z]

    @cast ll[x, y, z, l1] := L_d[x, y, z] * lel1[z, l1]
    @tensor LL[x, y, l1, l2] := ll[x, y, z, l1] * lel2[z, l2] # D x D x 2^12 x 2^6

    @cast lu[x, y, z, u1] := B_d[x, y, z] * leu1[z, u1]
    @tensor BB[x, y, u1, u2] := lu[x, y, z, u1] * leu2[z, u2] # D x D x 2^12 x 2^6

    LL = LL[:, :, p1l, p2l]
    @cast LL[x, y, (s, r)] := LL[x, y, s, r]

    BB = BB[:, :, p1u, p2u]
    @cast BB[x, y, (s, r)] := BB[x, y, s, r]

    RR = repeat(R_d, outer=(1, 1, size(pd, 1), 1))
    @cast RR[x, y, (z, σ)] := RR[x, y, z, σ] (σ ∈ 1:size(pd, 1))

    Anew_no_le = LL ⊠ BB ⊠ RR

    An = Anew_no_le .* reshape(CUDA.CuArray(M.loc_exp), 1, 1, :)
    @reduce Anew[x, y, z] := sum(σ) An[x, y, (z, σ)] (σ ∈ 1:length(pr))

    @tensor ret[x, y, d] := Anew[x, y, z] * ipd[z, d]

    Array(permutedims(ret, (1, 3, 2)) ./ maximum(abs.(ret)))
end