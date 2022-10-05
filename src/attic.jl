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