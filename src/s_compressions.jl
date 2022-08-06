using SparseArrays

export
    variational_compress!,
    _left_nbrs_site,
    _right_nbrs_site,
    compress_twosite!,
    canonise_truncate!,
    variational_sweep!,
    Environment,
    optimize_gauges_for_overlaps!!

"""
$(TYPEDSIGNATURES)
"""
mutable struct Environment <: AbstractEnvironment
    bra::QMps  # to be optimized
    mpo::QMpo
    ket::QMps
    trans::Symbol
    env::Dict

    function Environment(
        bra::QMps,
        mpo::QMpo,
        ket::QMps,
        trans::Symbol=:n
    )
        @assert trans ∈ (:n, :c)
        @assert bra.sites == ket.sites
        @assert issubset(bra.sites, mpo.sites)

        env0 = Dict(
            (first(bra.sites), :left) => ones(1, 1, 1),
            (last(bra.sites), :right) => ones(1, 1, 1)
        )
        env = new(bra, mpo, ket, trans, env0)
        for site ∈ env.bra.sites update_env_left!(env, site, trans) end
        env
    end
end

"""
$(TYPEDSIGNATURES)
"""
function variational_compress!(
    bra::QMps,
    mpo::QMpo,
    ket::QMps,
    tol::Real=1E-10,
    max_sweeps::Int=4,
    trans::Symbol=:n,
    args...
)
    env = Environment(bra, mpo, ket, trans)
    overlap = Inf
    overlap_before = measure_env(env, last(env.bra.sites), trans)

    for sweep ∈ 1:max_sweeps
        _left_sweep_var!(env, trans, args...)
        _right_sweep_var!(env, trans, args...)

        overlap = measure_env(env, last(env.bra.sites), trans)
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

"""
$(TYPEDSIGNATURES)
"""
function _left_sweep_var!(env::Environment, trans::Symbol=:n, args...)
    for site ∈ reverse(env.bra.sites)
        update_env_right!(env, site, trans)
        A = project_ket_on_bra(env, site, trans)
        @cast B[x, (y, z)] := A[x, y, z]
        _, Q = rq_fact(B, args...)
        @cast C[x, σ, y] := Q[x, (σ, y)] (σ ∈ 1:size(A, 2))
        env.bra[site] = C
        clear_env_containing_site!(env, site)
    end
end

"""
$(TYPEDSIGNATURES)
"""
function _right_sweep_var!(env::Environment, trans::Symbol=:n, args...)
    for site ∈ env.bra.sites
        update_env_left!(env, site, trans)
        A = project_ket_on_bra(env, site, trans)
        @cast B[(x, y), z] := A[x, y, z]
        Q, _ = qr_fact(B, args...)
        @cast C[x, σ, y] := Q[(x, σ), y] (σ ∈ 1:size(A, 2))
        env.bra[site] = C
        clear_env_containing_site!(env, site)
    end
end

"""
$(TYPEDSIGNATURES)
Largest x in sites: x < site
"""
function _left_nbrs_site(site::Site, sites)
    ls = filter(i -> i < site, sites)
    if isempty(ls) return -Inf end
    maximum(ls)
end

"""
$(TYPEDSIGNATURES)
Smallest x in sites: x > site
"""
function _right_nbrs_site(site::Site, sites)
    ms = filter(i -> i > site, sites)
    if isempty(ms) return Inf end
    minimum(ms)
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_left!(env::Environment, site::Site, trans::Symbol=:n)
    if site <= first(env.bra.sites) return end

    ls = _left_nbrs_site(site, env.bra.sites)
    LL = update_env_left(env.env[(ls, :left)], env.bra[ls], env.mpo[ls], env.ket[ls], trans)

    rs = _right_nbrs_site(ls, env.mpo.sites)
    while rs < site
        LL = update_env_left(LL, env.mpo[rs], trans)
        rs = _right_nbrs_site(rs, env.mpo.sites)
    end
    push!(env.env, (site, :left) => LL)
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_right!(env::Environment, site::Site, trans::Symbol=:n)
    if site >= last(env.bra.sites) return end

    rs = _right_nbrs_site(site, env.bra.sites)
    RR = update_env_right(env.env[(rs, :right)], env.bra[rs], env.mpo[rs], env.ket[rs], trans)

    ls = _left_nbrs_site(rs, env.mpo.sites)
    while ls > site
        RR = update_env_right(RR, env.mpo[ls], trans)
        ls = _left_nbrs_site(ls, env.mpo.sites)
    end
    push!(env.env, (site, :right) => RR)
end

"""
$(TYPEDSIGNATURES)
"""
function clear_env_containing_site!(env::Environment, site::Site)
    delete!(env.env, (_left_nbrs_site(site, env.ket.sites), :right))
    delete!(env.env, (_right_nbrs_site(site, env.ket.sites), :left))
end

"""
$(TYPEDSIGNATURES)
        -- A --
      |    |
 L = LE -- M --
      |    |
        -- B --
"""
function update_env_left(
    LE::S, A₀::S, M::T, B₀::S, trans::Symbol=:n
) where {S <: AbstractArray{Float64, 3}, T <: AbstractDict}
    sites = sort(collect(keys(M)))
    A = _update_tensor_forward(A₀, M, sites, Val(trans))
    B = _update_tensor_backwards(B₀, M, sites, Val(trans))
    update_env_left(LE, A, M[0], B, Val(trans))
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_left(
    LE::S, M::T, trans::Symbol=:n
) where {S <: AbstractArray{Float64, 3}, T <: AbstractDict}
    update_env_left(LE, M[0], Val(trans))
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_left(
    LE::S, M::T, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: AbstractArray{Float64, 2}}
    @tensor L[nt, nc, nb] :=  LE[nt, oc, nb] * M[oc, nc]
    L
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_left(
    LE::S, M::T, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: SparseCentralTensor}
    M11 = M.e11
    M12 = M.e12
    M21 = M.e21
    M22 = M.e22

    @cast MM[(l1, l2), (r1, r2)] := M11[l1,r1] * M21[l2, r1] * M12[l1, r2] * M22[l2, r2]
    @tensor L[nt, nc, nb] :=  LE[nt, oc, nb] * MM[oc, nc]
    L
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_left(
    LE::S, M::T, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: AbstractArray{Float64, 2}}
    @tensor L[nt, nc, nb] :=  LE[nt, oc, nb] * M[oc, nc]
    L
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_left(
    LE::S, M::T, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: SparseCentralTensor}
    M11 = M.e11
    M12 = M.e12
    M21 = M.e21
    M22 = M.e22

    @cast MM[(l1, l2), (r1, r2)] := M11[l1,r1] * M21[l2, r1] * M12[l1, r2] * M22[l2, r2]
    @tensor L[nt, nc, nb] :=  LE[nt, oc, nb] * MM[oc, nc]
    L
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_left(
    LE::S, A::S, M::T, B::S, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: AbstractArray{Float64, 4}}
    @tensor L[nb, nc, nt] := LE[ob, oc, ot] * A[ot, α, nt] *
                             M[oc, α, nc, β] * B[ob, β, nb] order = (ot, α, oc, β, ob)
    L
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_left(
    LE::S, A::S, M::T, B::S, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: AbstractArray{Float64, 4}}
    @tensor L[nb, nc, nt] := LE[ob, oc, ot] * A[ot, α, nt] *
                             M[oc, β, nc, α] * B[ob, β, nb] order = (ot, α, oc, β, ob)
    L
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_left(
    LE::S, A::S, M::T, B::S, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: SparseSiteTensor}
    L = CUDA.zeros(eltype(LE), size(B, 3), size(A, 3), maximum(M.projs[3]))

    A_d = permutedims(CUDA.CuArray(A[:, M.projs[2], :]), (1, 3, 2))
    L_d = permutedims(CUDA.CuArray(LE[:, M.projs[1], :]), (1, 3, 2))
    B_d = permutedims(CUDA.CuArray(B[:, M.projs[4], :]), (3, 1, 2))

    Lr_d = B_d ⊠ L_d ⊠ A_d
    Lr_d .*= reshape(CUDA.CuArray(M.loc_exp), 1, 1, :)

    pr = M.projs[3]

    println("update_env_left ::Val{:n}")
    @time begin
        # This is how sparse matrix is represented internally  
        csrRowPtr = CuArray(collect(1:length(pr) + 1))
        csrColInd = CuArray(pr)
        csrNzVal = CUDA.ones(Float64, length(pr))
        ipr = CUSPARSE.CuSparseMatrixCSC(csrRowPtr, csrColInd, csrNzVal, (maximum(pr), length(pr))) # transposed right here
        Lr_d = permutedims(Lr_d, (3, 2, 1)) # change dims to make matrix multiplication
        x, y, z = size(Lr_d)
        @cast Lr_d[x, (y, z)] := Lr_d[x, y, z]

        L = ipr * Lr_d  
        L = reshape(L, (:, y, z))
    end
    Array(permutedims(L, (3, 1, 2)))

end

#TODO: This implementation may not be optimal as is not batching matrix multiplication.
#=
"""
$(TYPEDSIGNATURES)
"""
function update_env_left(
     LE::S, A::S, M::T, B::S, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: SparseSiteTensor}
    Ap = permutedims(A, (1, 3, 2))
    LEp = permutedims(LE, (1, 3, 2))
    Bp = permutedims(B, (3, 1, 2))

    L = zeros(size(B, 3), size(A, 3), maximum(M.projs[3]))
    for (σ, lexp) ∈ enumerate(M.loc_exp)
        AA = @inbounds @view Ap[:, :, M.projs[2][σ]]
        LL = @inbounds @view LEp[:, :, M.projs[1][σ]]
        BB = @inbounds @view Bp[:, :, M.projs[4][σ]]
        @inbounds L[:, :, M.projs[3][σ]] += lexp .* (BB * LL * AA)
     end
     permutedims(L, (1, 3, 2))
end
=#

"""
$(TYPEDSIGNATURES)
"""
function update_env_left(
    L::S, A::S, M::T, B::S, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: SparsePegasusSquareTensor}
    pr, pd = M.projs
    p1l, p2l, p1u, p2u = M.bnd_projs

    @time ipr = cuIdentity(eltype(L), maximum(pr))[pr, :]
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

# """
# $(TYPEDSIGNATURES)
# """
# function update_env_left(
#     LE::S, A::S, M::T, B::S, ::Val{:c}
# ) where {S <: AbstractArray{Float64, 3}, T <: SparseSiteTensor}
#     L = zeros(size(B, 3), maximum(M.projs[3]), size(A, 3))

#     for (σ, lexp) ∈ enumerate(M.loc_exp)
#         AA = @inbounds @view A[:, M.projs[4][σ], :]
#         LL = @inbounds @view LE[:, M.projs[1][σ], :]
#         BB = @inbounds @view B[:, M.projs[2][σ], :]
#         @inbounds L[:, M.projs[3][σ], :] += lexp .* (BB' * LL * AA)
#     end
#     L ./ maximum(abs.(L))
# end

function update_env_left(
    LE::S, A::S, M::T, B::S, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: SparseSiteTensor}
    L = CUDA.zeros(eltype(LE), size(B, 3), size(A, 3), maximum(M.projs[3]))

    A_d = permutedims(CUDA.CuArray(A[:, M.projs[4], :]), (1, 3, 2))
    L_d = permutedims(CUDA.CuArray(LE[:, M.projs[1], :]), (1, 3, 2))
    B_d = permutedims(CUDA.CuArray(B[:, M.projs[2], :]), (3, 1, 2))

    Lr_d = B_d ⊠ L_d ⊠ A_d
    Lr_d .*= reshape(CUDA.CuArray(M.loc_exp), 1, 1, :)

    pr = M.projs[3]

    # for i in 1:maximum(pr)
    #     L[:,:,i] = sum(Lr_d[:, :, pr.==i], dims=3)
    # end

    println("update_env_left ::Val{:c}")
    @time begin
        csrRowPtr = CuArray(collect(1:length(pr) + 1))
        csrColInd = CuArray(pr)
        csrNzVal = CUDA.ones(Float64, length(pr))
        ipr = CUSPARSE.CuSparseMatrixCSC(csrRowPtr, csrColInd, csrNzVal, (maximum(pr), length(pr))) # transposed right here

        Lr_d = permutedims(Lr_d, (3, 2, 1)) #(256, 4, 4)
        _, y, z = size(Lr_d)
        @cast Lr_d[x, (y, z)] := Lr_d[x, y, z]

        L = ipr * Lr_d  #(16, 16)
        L = reshape(L, (:, y, z))
    end

    Array(permutedims(L, (3, 1, 2)) ./ maximum(abs.(L)))
    # Array(permutedims(L, (1, 3, 2)) ./ maximum(abs.(L)))

end

"""
$(TYPEDSIGNATURES)
"""
function update_env_left(
    L::S, A::S, M::T, B::S, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: SparsePegasusSquareTensor}
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
function update_env_left(
    LE::S, A::S, M::T, B::S, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    @cast A4[x, k, l, y] := A[x, (k, l), y] (k ∈ 1:maximum(p_rt))
    @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lb))

    L = zeros(size(B, 3), length(p_r), size(A, 3))
    for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
        AA = @inbounds @view A4[:, p_rt[r], p_lt[l], :]
        LL = @inbounds @view LE[:, l, :]
        BB = @inbounds @view B4[:, p_lb[l], p_rb[r], :]
        @inbounds L[:, r, :] += h[p_l[l], p_r[r]] .* (BB' * LL * AA)
    end
    L
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_left(
    LE::S, A::S, M::T, B::S, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    @cast A4[x, k, l, y] := A[x, (k, l), y] (k ∈ 1:maximum(p_rt))
    @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lb))

    L = zeros(size(B, 3), length(p_r), size(A, 3))
    for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
        AA = @inbounds @view A4[:, p_lb[l], p_rb[r], :]
        LL = @inbounds @view LE[:, l, :]
        BB = @inbounds @view B4[:, p_rt[r], p_lt[l], :]
        @inbounds L[:, r, :] += h[p_l[l], p_r[r]] .* (BB' * LL * AA)
    end
    L
end

"""
$(TYPEDSIGNATURES)
"""
function _update_tensor_forward(
    A::S, M::T, sites, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: AbstractDict}
    B = copy(A)
    for i ∈ sites
        if i == 0 break end
        C = M[i]
        B = _update_tensor_forward_n(C, B)
    end
    B
end

function _update_tensor_forward_n(
    C::T, B::S
    ) where {S <: AbstractArray{Float64, 3}, T <: AbstractArray{Float64, 2}}
    @tensor B[l, x, r] := B[l, y, r] * C[y, x]
    B
end

function _update_tensor_forward_n(
    C::T, B::S
    ) where {S <: AbstractArray{Float64, 3}, T <: SparseCentralTensor}
    M11 = C.e11
    M12 = C.e12
    M21 = C.e21
    M22 = C.e22

    @cast MM[(l1, l2), (r1, r2)] := M11[l1,r1] * M21[l2, r1] * M12[l1, r2] * M22[l2, r2]
    @tensor B[l, x, r] := B[l, y, r] * MM[y, x]
    B
end

"""
$(TYPEDSIGNATURES)
"""
function _update_tensor_forward(
    A::S, M::T, sites, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: AbstractDict}
    B = copy(A)
    for i ∈ reverse(sites)
        if i == 0 break end
        C = M[i]
        B = _update_tensor_forward_c(C, B)
    end
    B
end

function _update_tensor_forward_c(
    C::T, B::S
    ) where {S <: AbstractArray{Float64, 3}, T <: AbstractArray{Float64, 2}}
    @tensor B[l, x, r] := B[l, y, r] * C[x, y]
    B
end

function _update_tensor_forward_c(
    C::T, B::S
    ) where {S <: AbstractArray{Float64, 3}, T <: SparseCentralTensor}
    M11 = C.e11
    M12 = C.e12
    M21 = C.e21
    M22 = C.e22

    @cast MM[(l1, l2), (r1, r2)] := M11[l1,r1] * M21[l2, r1] * M12[l1, r2] * M22[l2, r2]
    @tensor B[l, x, r] := B[l, y, r] * MM[x, y]
    B
end

"""
$(TYPEDSIGNATURES)
"""
function _update_tensor_backwards(
    A::S, M::T, sites, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: AbstractDict}
    B = copy(A)
    for i ∈ reverse(sites)
        if i == 0 break end
        C = M[i]
        B = _update_tensor_backwards_n(C, B)
    end
    B
end

function _update_tensor_backwards_n(
    C::T, B::S
    ) where {S <: AbstractArray{Float64, 3}, T <: AbstractArray{Float64, 2}}
    @tensor B[l, x, r] := B[l, y, r] * C[x, y]
end

function _update_tensor_backwards_n(
    C::T, B::S
    ) where {S <: AbstractArray{Float64, 3}, T <: SparseCentralTensor}
    M11 = C.e11
    M12 = C.e12
    M21 = C.e21
    M22 = C.e22

    @cast MM[(l1, l2), (r1, r2)] := M11[l1,r1] * M21[l2, r1] * M12[l1, r2] * M22[l2, r2]
    @tensor B[l, x, r] := B[l, y, r] * MM[x, y]
end

"""
$(TYPEDSIGNATURES)
"""
function _update_tensor_backwards(
    A::S, M::T, sites, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: AbstractDict}
    B = copy(A)
    for i ∈ sites
        if i == 0 break end
        C = M[i]
        B = _update_tensor_backwards_c(C, B)
    end
    B
end

function _update_tensor_backwards_c(
    C::T, B::S
    ) where {S <: AbstractArray{Float64, 3}, T <: AbstractArray{Float64, 2}}
    @tensor B[l, x, r] := B[l, y, r] * C[y, x]
end

function _update_tensor_backwards_c(
    C::T, B::S
    ) where {S <: AbstractArray{Float64, 3}, T <: SparseCentralTensor}
    M11 = C.e11
    M12 = C.e12
    M21 = C.e21
    M22 = C.e22

    @cast MM[(l1, l2), (r1, r2)] := M11[l1,r1] * M21[l2, r1] * M12[l1, r2] * M22[l2, r2]
    @tensor B[l, x, r] := B[l, y, r] * MM[y, x]
end

"""
$(TYPEDSIGNATURES)
      -- A --
         |    |
 R =  -- M -- RE
         |    |
      -- B --
"""
function update_env_right(
    RE::S, A::S, M::T, B::S, ::Val{:n}
) where {T <: AbstractArray{Float64, 4}, S <: AbstractArray{Float64, 3}}
    @tensor R[nt, nc, nb] := RE[ot, oc, ob] * A[nt, α, ot] *
                             M[nc, α, oc, β] * B[nb, β, ob] order = (ot, α, oc, β, ob)
    R
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_right(
    RE::S, A::S, M::T, B::S, ::Val{:c}
) where {T <: AbstractArray{Float64, 4}, S <: AbstractArray{Float64, 3}}
    @tensor R[nt, nc, nb] := RE[ot, oc, ob] * A[nt, α, ot] *
                             M[nc, β, oc, α] * B[nb, β, ob] order = (ot, α, oc, β, ob)
    R
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_right(
    RE::S, A::S, M::T, B::S, ::Val{:n}
) where {T <: SparseSiteTensor, S <: AbstractArray{Float64, 3}}
    R = CUDA.zeros(eltype(RE), size(A, 1), size(B, 1), maximum(M.projs[1]))

    A_d = permutedims(CUDA.CuArray(A[:, M.projs[2], :]), (1, 3, 2))
    R_d = permutedims(CUDA.CuArray(RE[:, M.projs[3], :]), (1, 3, 2))
    B_d = permutedims(CUDA.CuArray(B[:, M.projs[4], :]), (3, 1, 2))

    Rr_d = A_d ⊠ R_d ⊠ B_d

    Rr_d .*= reshape(CUDA.CuArray(M.loc_exp), 1, 1, :)
    pr = M.projs[1]

    println("update_env_right ::Val{:n}")
    @time begin
        csrRowPtr = CuArray(collect(1:length(pr) + 1))
        csrColInd = CuArray(pr)
        csrNzVal = CUDA.ones(Float64, length(pr))
        ipr = CUSPARSE.CuSparseMatrixCSC(csrRowPtr, csrColInd, csrNzVal, (maximum(pr), length(pr))) # transposed right here

        Rr_d = permutedims(Rr_d, (3, 2, 1)) 
        _, y, z = size(Rr_d)
        @cast Rr_d[x, (y, z)] := Rr_d[x, y, z]

        R = ipr * Rr_d  
        R = reshape(R, (:, y, z))
    end
    
    # for i in 1:maximum(pr)
    #     R[:,:,i] = sum(Rr_d[:, :, pr.==i], dims=3)
    # end
    
    Array(permutedims(R, (3, 1, 2)))
    # Array(permutedims(R, (1, 3, 2)))
end

# function update_env_right(
#     RE::S, A::S, M::T, B::S, ::Val{:n}
# ) where {T <: SparseSiteTensor, S <: AbstractArray{Float64, 3}}
#     R = zeros(size(A, 1), maximum(M.projs[1]), size(B, 1))

#     for (σ, lexp) ∈ enumerate(M.loc_exp)
#         AA = @inbounds @view A[:, M.projs[2][σ], :]
#         RR = @inbounds @view RE[:, M.projs[3][σ], :]
#         BB = @inbounds @view B[:, M.projs[4][σ], :]
#         @inbounds R[:, M.projs[1][σ], :] += lexp .* (AA * RR * BB')
#     end
#     R
# end

"""
$(TYPEDSIGNATURES)
"""
function update_env_right(
    R::S, A::S, M::T, B::S, ::Val{:n}
) where {T <: SparsePegasusSquareTensor, S <: AbstractArray{Float64, 3}}
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
    RE::S, A::S, M::T, B::S, ::Val{:c}
) where {T <: SparseSiteTensor, S <: AbstractArray{Float64, 3}}
    R = CUDA.zeros(eltype(RE), size(A, 1), size(B, 1), maximum(M.projs[1]))

    A_d = permutedims(CUDA.CuArray(A[:, M.projs[4], :]), (1, 3, 2))
    R_d = permutedims(CUDA.CuArray(RE[:, M.projs[3], :]), (1, 3, 2))
    B_d = permutedims(CUDA.CuArray(B[:, M.projs[2], :]), (3, 1, 2))

    Rr_d = A_d ⊠ R_d ⊠ B_d

    Rr_d .*= reshape(CUDA.CuArray(M.loc_exp), 1, 1, :)
    pr = M.projs[1]

    println("update_env_left ::Val{:c}")
    @time begin
        csrRowPtr = CuArray(collect(1:length(pr) + 1))
        csrColInd = CuArray(pr)
        csrNzVal = CUDA.ones(Float64, length(pr))
        ipr = CUSPARSE.CuSparseMatrixCSC(csrRowPtr, csrColInd, csrNzVal, (maximum(pr), length(pr))) # transposed right here

        Rr_d = permutedims(Rr_d, (3, 2, 1)) #(256, 4, 4)
        _, y, z = size(Rr_d)
        @cast Rr_d[x, (y, z)] := Rr_d[x, y, z]

        R = ipr * Rr_d  #(16, 16)
        R = reshape(R, (:, y, z))
    end

    
    # for i in 1:maximum(pr)
    #     R[:,:,i] = sum(Rr_d[:, :, pr.==i], dims=3)
    # end
    
    Array(permutedims(R, (3, 1, 2)))
    # Array(permutedims(R, (1, 3, 2)))
end

# function update_env_right(
#     RE::S, A::S, M::T, B::S, ::Val{:c}
# ) where {T <: SparseSiteTensor, S <: AbstractArray{Float64, 3}}
#     R = zeros(size(A, 1), maximum(M.projs[1]), size(B, 1))

#     for (σ, lexp) ∈ enumerate(M.loc_exp)
#         AA = @inbounds @view A[:, M.projs[4][σ], :]
#         RR = @inbounds @view RE[:, M.projs[3][σ], :]
#         BB = @inbounds @view B[:, M.projs[2][σ], :]
#         @inbounds R[:, M.projs[1][σ], :] += lexp .* (AA * RR * BB')
#     end
#     R
# end


"""
$(TYPEDSIGNATURES)
"""
function update_env_right(
    R::S, A::S, M::T, B::S, ::Val{:c}
) where {T <: SparsePegasusSquareTensor, S <: AbstractArray{Float64, 3}}
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


"""
$(TYPEDSIGNATURES)
"""
function update_env_right(
    RE::S, A::S, M::T, B::S, ::Val{:n}
) where {T <: SparseVirtualTensor, S <: AbstractArray{Float64,3}}
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs
    @cast A4[x, k, l, y] := A[x, (k, l), y] (k ∈ 1:maximum(p_rt))
    @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lb))

    R = zeros(size(A, 1), length(p_l), size(B, 1))
    for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
        AA = @inbounds @view A4[:, p_rt[r], p_lt[l], :]
        RR = @inbounds @view RE[:, r, :]
        BB = @inbounds @view B4[:, p_lb[l], p_rb[r], :]
        @inbounds R[:, l, :] += h[p_l[l], p_r[r]] * (AA * RR * BB')
    end
    R
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_right(
    RE::S, A::S, M::T, B::S, ::Val{:c}
) where {T <: SparseVirtualTensor, S <: AbstractArray{Float64, 3}}
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs
    @cast A4[x, k, l, y] := A[x, (k, l), y] (k ∈ 1:maximum(p_rt))
    @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lb))

    R = zeros(size(A, 1), length(p_l), size(B, 1))
    for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
        AA = @inbounds @view A4[:, p_lb[l], p_rb[r], :]
        RR = @inbounds @view RE[:, r, :]
        BB = @inbounds @view B4[:, p_rt[r], p_lt[l], :]
        @inbounds R[:, l, :] += h[p_l[l], p_r[r]] * (AA * RR * BB')
    end
    R
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_right(
    RE::S, A₀::S1, M::T, B₀::S, trans::Symbol=:n
) where {T <: AbstractDict, S <: AbstractArray{Float64, 3}, S1 <: AbstractArray{Float64, 3}}
    sites = sort(collect(keys(M)))
    A = _update_tensor_forward(A₀, M, sites, Val(trans))
    B = _update_tensor_backwards(B₀, M, sites, Val(trans))
    update_env_right(RE, A, M[0], B, Val(trans))
end

"""
$(TYPEDSIGNATURES)
           --
              |
 R =  -- M -- RE
              |
           --
"""
function update_env_right(
    RE::S, M::T, trans::Symbol=:c
) where {S <: AbstractArray{Float64, 3}, T <: AbstractDict}
    update_env_right(RE, M[0], Val(trans))
end

function update_env_right(
    RE::S, M::T, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: AbstractArray{Float64, 2}}
    @tensor R[nt, nc, nb] := M[nc, oc] * RE[nt, oc, nb]
    R
end

function update_env_right(
    RE::S, M::T, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: SparseCentralTensor}
    M11 = M.e11
    M12 = M.e12
    M21 = M.e21
    M22 = M.e22

    @cast MM[(l1, l2), (r1, r2)] := M11[l1,r1] * M21[l2, r1] * M12[l1, r2] * M22[l2, r2]

    @tensor R[nt, nc, nb] := MM[nc, oc] * RE[nt, oc, nb]
    R
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_right(
    RE::S, M::T, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: AbstractArray{Float64, 2}}
    @tensor R[nt, nc, nb] := M[nc, oc] * RE[nt, oc, nb]
    R
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_right(
    RE::S, M::T, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: SparseCentralTensor}
    M11 = M.e11
    M12 = M.e12
    M21 = M.e21
    M22 = M.e22

    @cast MM[(l1, l2), (r1, r2)] := M11[l1,r1] * M21[l2, r1] * M12[l1, r2] * M22[l2, r2]

    @tensor R[nt, nc, nb] := MM[nc, oc] * RE[nt, oc, nb]
    R
end

"""
$(TYPEDSIGNATURES)
"""
function project_ket_on_bra(env::Environment, site::Site, trans::Symbol=:n)
    project_ket_on_bra(
        env.env[(site, :left)],
        env.ket[site],
        env.mpo[site],
        env.env[(site, :right)],
        Val(trans)
    )
end

"""
$(TYPEDSIGNATURES)
"""
function project_ket_on_bra_twosite(env::Environment, site::Site)
    site_l = _left_nbrs_site(site, env.bra.sites)
    project_ket_on_bra(
        env.env[(site_l, :left)],
        env.ket[site_l],
        env.ket[site],
        env.mpo[site_l][0],
        env.mpo[site][0],
        env.env[(site, :right)]
    )
end

"""
$(TYPEDSIGNATURES)
   |    |    |
  LE -- M -- RE
   |    |    |
     -- B --
"""
function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{:n}
) where {T <: AbstractArray{Float64, 4}, S <: AbstractArray{Float64, 3}}
    @tensor A[x, y, z] := LE[k, l, x] * B[k, m, o] *
                          M[l, y, n, m] * RE[z, n, o] order = (k, l, m, n, o)
    A
end

"""
$(TYPEDSIGNATURES)
"""
function project_ket_on_bra(
    LE::S, B::S, C::S, M::T, N::T, RE::S, ::Val{:n}
) where {T <: AbstractArray{Float64, 4}, S <: AbstractArray{Float64, 3}}
    @tensor A[x, y, z, r] := LE[k, l, x] * B[k, m, o] *
                          M[l, y, n, m] * C[o, s, q] *
                          N[n, z, p, s] * RE[r, p, q] order = (k, l, m, n, o, s, p, q)
    A
end

"""
$(TYPEDSIGNATURES)
"""
function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{:n}
) where {T <: AbstractArray{Float64, 2}, S <: AbstractArray{Float64, 3}}
    @tensor A[x, y, z] := B[x, a, z] * M[y, a]
    A
end

"""
$(TYPEDSIGNATURES)
"""
function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{:c}
) where {T <: AbstractArray{Float64, 2}, S <: AbstractArray{Float64, 3}}
    @tensor A[x, y, z] := B[x, a, z] * M[a, y]
    A
end

"""
$(TYPEDSIGNATURES)
"""
function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{:n}
) where {T <: SparseCentralTensor, S <: AbstractArray{Float64, 3}}
    M11 = M.e11
    M12 = M.e12
    M21 = M.e21
    M22 = M.e22

    @cast MM[(l1, l2), (r1, r2)] := M11[l1,r1] * M21[l2, r1] * M12[l1, r2] * M22[l2, r2]

    @tensor A[x, y, z] := B[x, a, z] * MM[y, a]
    A
end

"""
$(TYPEDSIGNATURES)
"""
function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{:c}
) where {T <: SparseCentralTensor, S <: AbstractArray{Float64, 3}}
    M11 = M.e11
    M12 = M.e12
    M21 = M.e21
    M22 = M.e22

    @cast MM[(l1, l2), (r1, r2)] := M11[l1,r1] * M21[l2, r1] * M12[l1, r2] * M22[l2, r2]

    @tensor A[x, y, z] := B[x, a, z] * MM[a, y]
    A
end

"""
$(TYPEDSIGNATURES)
"""
function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: SparseSiteTensor}
    A = CUDA.zeros(eltype(LE), size(LE, 3), size(RE, 1), maximum(M.projs[2]))

    le = permutedims(CUDA.CuArray(LE[:, M.projs[1], :]), (3, 1, 2))
    b = permutedims(CUDA.CuArray(B[:, M.projs[4], :]), (1, 3, 2))
    re = permutedims(CUDA.CuArray(RE[:, M.projs[3], :]), (3, 1, 2))

    Ar_d = le ⊠ b ⊠ re
    Ar_d .*= reshape(CUDA.CuArray(M.loc_exp), 1, 1, :)

    pu = M.projs[2]

    # for i in 1:maximum(pu)
    #     A[:,:,i] = sum(Ar_d[:, :, pu.==i], dims=3)
    # end
    # Array(permutedims(A, (1, 3, 2)))

    println("project_ket_on_bra ::Val{:n}")
    @time begin
        csrRowPtr = CuArray(collect(1:length(pu) + 1))
        csrColInd = CuArray(pu)
        csrNzVal = CUDA.ones(Float64, length(pu))
        ipu = CUSPARSE.CuSparseMatrixCSC(csrRowPtr, csrColInd, csrNzVal, (maximum(pu), length(pu))) # transposed right here

        Ar_d = permutedims(Ar_d, (3, 2, 1)) #(256, 4, 4)
        _, y, z = size(Ar_d)
        @cast Ar_d[x, (y, z)] := Ar_d[x, y, z]

        A = ipu * Ar_d  #(16, 16)
        A = reshape(A, (:, y, z))
    end
    
    Array(permutedims(A, (3, 1, 2)))
end

# function project_ket_on_bra(
#     LE::S, B::S, M::T, RE::S, ::Val{:n}
# ) where {S <: AbstractArray{Float64, 3}, T <: SparseSiteTensor}
#     A = zeros(size(LE, 3), maximum(M.projs[2]), size(RE, 1))

#     for (σ, lexp) ∈ enumerate(M.loc_exp)
#         le = @inbounds @view LE[:, M.projs[1][σ], :]
#         b = @inbounds @view B[:, M.projs[4][σ], :]
#         re = @inbounds @view RE[:, M.projs[3][σ], :]
#         @inbounds A[:, M.projs[2][σ], :] += lexp .* (le' * b * re')
#     end
#     A
# end



function project_ket_on_bra(
    L::S, B::S, M::T, R::S, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: SparsePegasusSquareTensor}
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
    LE::S, B::S, M::T, RE::S, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs
    @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lb))

    A = zeros(size(LE, 3), maximum(p_rt), maximum(p_lt), size(RE, 1))
    for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
        le = @inbounds @view LE[:, l, :]
        b = @inbounds @view B4[:, p_lb[l], p_rb[r], :]
        re = @inbounds @view RE[:, r, :]
        @inbounds A[:,  p_rt[r], p_lt[l], :] += h[p_l[l], p_r[r]] .* (le' * b * re')
    end
    @cast AA[l, (ũ, u), r] := A[l, ũ, u, r]
    AA
end

"""
$(TYPEDSIGNATURES)
"""
function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{:c}
) where {T <: AbstractArray{Float64, 4}, S <: AbstractArray{Float64, 3}}
    @tensor A[x, y, z] := LE[k, l, x] * B[k, m, o] *
                          M[l, m, n, y] * RE[z, n, o] order = (k, l, m, n, o)
    A
end

"""
$(TYPEDSIGNATURES)
"""
function project_ket_on_bra(
    LE::S, B::S, C::S, M::T, N::T, RE::S, ::Val{:c}
) where {T <: AbstractArray{Float64, 4}, S <: AbstractArray{Float64, 3}}
    @tensor A[x, m, s, r] := LE[k, l, x] * B[k, y, o] *
                          M[l, y, n, m] * C[o, z, q] *
                          N[n, z, p, s] * RE[r, p, q] order = (k, l, y, n, o, z, p, q)
    A
end

"""
$(TYPEDSIGNATURES)
"""
function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: SparseSiteTensor}
A = CUDA.zeros(eltype(LE), size(LE, 3), size(RE, 1), maximum(M.projs[4]))

le = permutedims(CUDA.CuArray(LE[:, M.projs[1], :]), (3, 1, 2))
b = permutedims(CUDA.CuArray(B[:, M.projs[2], :]), (1, 3, 2))
re = permutedims(CUDA.CuArray(RE[:, M.projs[3], :]), (3, 1, 2))

Ar_d = le ⊠ b ⊠ re
Ar_d .*= reshape(CUDA.CuArray(M.loc_exp), 1, 1, :)

pu = M.projs[4]

for i in 1:maximum(pu)
    A[:,:,i] = sum(Ar_d[:, :, pu.==i], dims=3)
end
Array(permutedims(A, (1, 3, 2)))
end

# function project_ket_on_bra(
#     LE::S, B::S, M::T, RE::S, ::Val{:c}
# ) where {S <: AbstractArray{Float64, 3}, T <: SparseSiteTensor}
#     A = zeros(size(LE, 3), maximum(M.projs[4]), size(RE, 1))

#     for (σ, lexp) ∈ enumerate(M.loc_exp)
#         le = @inbounds @view LE[:, M.projs[1][σ], :]
#         b = @inbounds @view B[:, M.projs[2][σ], :]
#         re = @inbounds @view RE[:, M.projs[3][σ], :]
#         @inbounds A[:, M.projs[4][σ], :] += lexp .* (le' * b * re')
#     end
#     A
# end

"""
$(TYPEDSIGNATURES)
"""
function project_ket_on_bra(
    L::S, B::S, M::T, R::S, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: SparsePegasusSquareTensor}
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


"""
$(TYPEDSIGNATURES)
"""
function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs
    @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lb))

    A = zeros(size(LE, 3), maximum(p_lb), maximum(p_rb), size(RE, 1))
    for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
        le = @inbounds @view LE[:, l, :]
        b = @inbounds @view B4[:, p_rt[r], p_lt[l], :]
        re = @inbounds @view RE[:, r, :]
        @inbounds  A[:, p_lb[l], p_rb[r], :] += h[p_l[l], p_r[r]] .* (le' * b * re')
    end
    @cast AA[l, (ũ, u), r] := A[l, ũ, u, r]
    AA
end

"""
$(TYPEDSIGNATURES)
"""
function project_ket_on_bra(
    LE::S, B₀::S, M::T, RE::S, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: AbstractDict}
    C = sort(collect(M), by = x -> x[1])
    TT = B₀
    for (_, v) ∈ reverse(C) TT = project_ket_on_bra(LE, TT, v, RE, Val(:n)) end
    TT
end

"""
$(TYPEDSIGNATURES)
"""
function project_ket_on_bra(
    LE::S, B₀::S, M::T, RE::S, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: AbstractDict}
    C = sort(collect(M), by = x -> x[1])
    TT = B₀
    for (_, v) ∈ C TT = project_ket_on_bra(LE, TT, v, RE, Val(:c)) end
    TT
end

"""
$(TYPEDSIGNATURES)
"""
function measure_env(env::Environment, site::Site, trans::Symbol=:n)
    L = update_env_left(
        env.env[(site, :left)], env.bra[site], env.mpo[site], env.ket[site], trans
    )
    R = env.env[(site, :right)]
    @tensor L[t, c, b] * R[b, c, t]
end

"""
$(TYPEDSIGNATURES)
"""
function truncate!(ψ::QMps, s::Symbol, Dcut::Int=typemax(Int), tolS::Real=1E-16, args...)
    @assert s ∈ (:left, :right)
    if s == :right
        _right_sweep!(ψ, args...)
        _left_sweep!(ψ, Dcut, tolS, args...)
    else
        _left_sweep!(ψ, args...)
        _right_sweep!(ψ, Dcut, tolS, args...)
    end
end

"""
$(TYPEDSIGNATURES)
"""
canonise!(ψ::QMps, s::Symbol) = canonise!(ψ, Val(s))

"""
$(TYPEDSIGNATURES)
"""
canonise!(ψ::QMps, ::Val{:right}) = _left_sweep!(ψ, typemax(Int))

"""
$(TYPEDSIGNATURES)
"""
canonise!(ψ::QMps, ::Val{:left}) = _right_sweep!(ψ, typemax(Int))

"""
$(TYPEDSIGNATURES)
"""
function variational_sweep!(
    bra::QMps,
    mpo::QMpo,
    ket::QMps,
    ::Val{:left},
    trans::Symbol=:n,
    args...
)
    env = Environment(bra, mpo, ket, trans)
    _right_sweep_var!(env, trans, args...)
end


"""
$(TYPEDSIGNATURES)
"""
function variational_sweep!(
    bra::QMps,
    mpo::QMpo,
    ket::QMps,
    ::Val{:right},
    trans::Symbol=:n,
    args...
)
    env = Environment(bra, mpo, ket, trans)
    _left_sweep_var!(env, trans, args...)
end

function canonise_truncate!(ψ::QMps, s::Symbol, Dcut::Int=typemax(Int), tolS::Real=1E-16, args...)
    @assert s ∈ (:left, :right)
    if s == :right
        _left_sweep!(ψ, Dcut, tolS, args...)
    else
        _right_sweep!(ψ, Dcut, tolS, args...)
    end
end

"""
$(TYPEDSIGNATURES)
"""
function _right_sweep!(ψ::QMps, Dcut::Int=typemax(Int), tolS::Real=1E-16, args...)
    R = ones(eltype(ψ[1]), 1, 1)
    for i ∈ ψ.sites
        A = ψ[i]
        @matmul M̃[(x, σ), y] := sum(α) R[x, α] * A[α, σ, y]
        Q, R = qr_fact(M̃, Dcut, tolS, args...)
        R ./= maximum(abs.(R))
        @cast A[x, σ, y] := Q[(x, σ), y] (σ ∈ 1:size(A, 2))
        ψ[i] = A
    end
end

"""
$(TYPEDSIGNATURES)
"""
function _left_sweep!(ψ::QMps, Dcut::Int=typemax(Int), tolS::Real=1E-16, args...)
    R = ones(eltype(ψ[1]), 1, 1)
    for i ∈ reverse(ψ.sites)
        B = ψ[i]
        @matmul M̃[x, (σ, y)] := sum(α) B[x, σ, α] * R[α, y]
        R, Q = rq_fact(M̃, Dcut, tolS, args...)
        R ./= maximum(abs.(R))
        @cast B[x, σ, y] := Q[x, (σ, y)] (σ ∈ 1:size(B, 2))
        ψ[i] = B
    end
end

"""
$(TYPEDSIGNATURES)
"""
function _gauges_right_sweep!!!(ψ_top::QMps, ψ_bot::QMps, all_gauges::Dict)
    RT, RB = ones(1, 1), ones(1, 1)
    for i ∈ ψ_top.sites
        T, B = ψ_top[i], ψ_bot[i]

        @tensor T[a, b, c] := RT[a, s] * T[s, b, c]
        @tensor B[a, b, c] := RB[a, s] * B[s, b, c]
        @tensor ρ_t[r, s] := T[i, r, j] * conj(T)[i, s, j]
        @tensor ρ_b[r, s] := B[i, r, j] * conj(B)[i, s, j]

        gauge = (diag(ρ_b) ./ diag(ρ_t)) .^ (1 / 4) # optimize
        gauge_inv = 1.0 ./ gauge
        all_gauges[i] .*= gauge # update

        AT = T .* reshape(gauge, (1, :, 1))
        AB = B .* reshape(gauge_inv, (1, :, 1))

        @cast ATR[(x, σ), y] := AT[x, σ, y]
        QT, RT = qr_fact(ATR)
        RT ./= maximum(abs.(RT))
        @cast AT[x, σ, y] := QT[(x, σ), y] (σ ∈ 1:size(AT, 2))
        ψ_top[i] = AT

        @cast ABR[(x, σ), y] := AB[x, σ, y]
        QB, RB = qr_fact(ABR)
        RB ./= maximum(abs.(RB))
        @cast AB[x, σ, y] := QB[(x, σ), y] (σ ∈ 1:size(AB, 2))
        ψ_bot[i] = AB
    end
end

"""
$(TYPEDSIGNATURES)
"""
function _gauges_left_sweep!!!(ψ_top::QMps, ψ_bot::QMps, all_gauges::Dict)
    RT, RB = ones(1, 1), ones(1, 1)

    for i ∈ reverse(ψ_top.sites)
        T, B = ψ_top[i], ψ_bot[i]

        @tensor T[a, b, c] := T[a, b, s] * RT[s, c]
        @tensor B[a, b, c] := B[a, b, s] * RB[s, c]
        @tensor ρ_t[r, s] := T[i, r, j] * conj(T)[i, s, j]
        @tensor ρ_b[r, s] := B[i, r, j] * conj(B)[i, s, j]

        gauge = (diag(ρ_b) ./ diag(ρ_t)) .^ (1 / 4) # optimize
        gauge_inv = 1.0 ./ gauge
        all_gauges[i] .*= gauge # update

        AT = T .* reshape(gauge, (1, :, 1))
        AB = B .* reshape(gauge_inv, (1, :, 1))

        @cast ATR[x, (σ, y)] := AT[x, σ, y]
        RT, QT = rq_fact(ATR)
        RT ./= maximum(abs.(RT))
        @cast AT[x, σ, y] := QT[x, (σ, y)] (σ ∈ 1:size(AT, 2))
        ψ_top[i] = AT

        @cast ABR[x, (σ, y)] := AB[x, σ, y]
        RB, QB = rq_fact(ABR)
        RB ./= maximum(abs.(RB))
        @cast AB[x, σ, y] := QB[x, (σ, y)] (σ ∈ 1:size(AB, 2))
        ψ_bot[i] = AB
    end
end

"""
$(TYPEDSIGNATURES)
"""
function optimize_gauges_for_overlaps!!(
    ψ_top::QMps,
    ψ_bot::QMps,
    tol::Real=1E-8,
    max_sweeps::Int=4
)
    canonise!(ψ_top, :right)
    canonise!(ψ_bot, :right)

    overlap_old = dot(ψ_top, ψ_bot)
    all_gauges = Dict(i => ones(size(ψ_top[i], 2)) for i ∈ ψ_top.sites)
    for _ ∈ 1:max_sweeps
        _gauges_right_sweep!!!(ψ_top, ψ_bot, all_gauges)
        _gauges_left_sweep!!!(ψ_top, ψ_bot, all_gauges)

        overlap_new = dot(ψ_top, ψ_bot)
        Δ = overlap_new / overlap_old
        overlap_old = overlap_new
        if abs(Δ - 1.0) < tol break end
    end
    all_gauges
end
