export compress!, _left_nbrs_site, _right_nbrs_site, compress_twosite!

mutable struct Environment <: AbstractEnvironment
    bra::QMPS  # to be optimized
    mpo::QMPO
    ket::QMPS
    trans::Symbol
    env::Dict

    function Environment(
        bra::QMPS,
        mpo::QMPO,
        ket::QMPS,
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

function SpinGlassTensors.compress!(
    bra::QMPS,
    mpo::QMPO,
    ket::QMPS,
    Dcut::Int,
    tol::Real=1E-8,
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

function compress_twosite!(
    bra::QMPS, mpo::QMPO, ket::QMPS, Dcut::Int, tol::Real=1E-8, max_sweeps::Int=4, args...
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
            env.bra[env.bra.sites[1]] = US/norm(US)
            update_env_right!(env, env.bra.sites[2])
            update_env_right!(env, env.bra.sites[1])
        end
    end
end

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

# Largest x in sites: x < site
function _left_nbrs_site(site::Site, sorted_sites)
    site_pos = searchsortedfirst(sorted_sites, site)
    site_pos == 1 ? -Inf : sorted_sites[site_pos-1]
end

# Smallest x in sites: x > site
function _right_nbrs_site(site::Site, sorted_sites)
    site_pos = searchsortedlast(sorted_sites, site)
    site_pos == length(sorted_sites) ? Inf : sorted_sites[site_pos+1]
end

function update_env_left!(env::Environment, site::Site, trans::Symbol=:n)
    if site <= first(env.bra.sites) return end

    ls = _left_nbrs_site(site, env.bra.sites)
    LL = update_env_left(env.env[(ls, :left)], env.bra[ls], env.mpo[ls], env.ket[ls], trans)

    rs = _right_nbrs_site(ls, env.mpo.sites)
    while rs < site
        @tensor LL[nt, nc, nb] :=  LE[nt, oc, nb] * env.mpo[rs][0][oc, nc]
        rs = _right_nbrs_site(rs, env.mpo.sites)
    end
    push!(env.env, (site, :left) => LL)
end

function update_env_right!(env::Environment, site::Site, trans::Symbol=:n)
    if site >= last(env.bra.sites) return end

    rs = _right_nbrs_site(site, env.bra.sites)
    RR = update_env_right(env.env[(rs, :right)], env.bra[rs], env.mpo[rs], env.ket[rs], trans)

    ls = _left_nbrs_site(rs, env.mpo.sites)
    while ls > site
        RR = update_env_right(RR, env.mpo[ls], Val(trans))
        ls = _left_nbrs_site(ls, env.mpo.sites)
    end
    push!(env.env, (site, :right) => RR)
end

function clear_env_containing_site!(env::Environment, site::Site)
    delete!(env.env, (_left_nbrs_site(site, env.ket.sites), :right))
    delete!(env.env, (_right_nbrs_site(site, env.ket.sites), :left))
end

#        -- A --
#      |    |
# L = LE -- M --
#      |    |
#        -- B --
function update_env_left(
    LE::AbstractArray{T, 3},
    A₀::AbstractArray{T, 3},
    M::Dict,
    B₀::AbstractArray{T, 3},
    trans::Symbol=:n
) where {T <: Real}
    sites = sort(collect(keys(M)))
    A =_update_tensor_forward(A₀, M, sites, Val(trans))
    B = _update_tensor_backwards(B₀, M, sites, Val(trans))
    B = trans == :c ? PermutedDimsArray(B, (1, 4, 3, 2)) : B
    @tensor L[nb, nc, nt] := LE[ob, oc, ot] * A[ot, α, nt] *
                             M[0][oc, α, nc, β] * B[ob, β, nb] order = (ot, α, oc, β, ob)
    L
end

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

function _update_tensor_forward(
    A::S, M::T, sites, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: AbstractDict}
    B = copy(A)
    for i ∈ sites
        if i == 0 break end
        C = M[i]
        @tensor B[l, x, r] := B[l, y, r] * C[y, x]
    end
    B
end

function _update_tensor_forward(
    A::S, M::T, sites, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: AbstractDict}
    B = copy(A)
    for i ∈ reverse(sites)
        if i == 0 break end
        C = M[i]
        @tensor B[l, x, r] := B[l, y, r] * C[x, y]
    end
    B
end

function _update_tensor_backwards(
    A::S, M::T, sites, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: AbstractDict}
    B = copy(A)
    for i ∈ reverse(sites)
        if i == 0 break end
        C = M[i]
        @tensor B[l, x, r] := B[l, y, r] * C[x, y]
    end
    B
end

function _update_tensor_backwards(
    A::S, M::T, sites, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: AbstractDict}
    B = copy(A)
    for i ∈ sites
        if i == 0 break end
        C = M[i]
        @tensor B[l, x, r] := B[l, y, r] * C[y, x]
    end
    B
end

#      -- A --
#         |    |
# R =  -- M -- RE
#         |    |
#      -- B --
function update_env_right(
    RE::S, A::S, M::T, B::S, ::Val{:n}
) where {T <: AbstractArray{Float64, 4}, S <: AbstractArray{Float64, 3}}
    @tensor R[nt, nc, nb] := RE[ot, oc, ob] * A[nt, α, ot] *
                             M[nc, α, oc, β] * B[nb, β, ob] order = (ot, α, oc, β, ob)
    R
end

function update_env_right(
    RE::S, A::S, M::T, B::S, ::Val{:c}
) where {T <: AbstractArray{Float64, 4}, S <: AbstractArray{Float64, 3}}
    @tensor R[nt, nc, nb] := RE[ot, oc, ob] * A[nt, α, ot] *
                             M[nc, β, oc, α] * B[nb, β, ob] order = (ot, α, oc, β, ob)
    R
end

function update_env_right(
    RE::S, A::S, M::T, B::S, ::Val{:n}
) where {T <: SparseSiteTensor, S} # {T <: SparseSiteTensor, S <: AbstractArray{Float64, 3}}
    R = zeros(size(A, 1), maximum(M.projs[1]), size(B, 1))

    #Threads.@threads for σ ∈ 1:length(M.loc_exp)
    #    lexp = M.loc_exp[σ]
    for (σ, lexp) ∈ enumerate(M.loc_exp)
        AA = @view A[:, M.projs[2][σ], :]
        RR = @view RE[:, M.projs[3][σ], :]
        BB = @view B[:, M.projs[4][σ], :]
        R[:, M.projs[1][σ], :] += lexp .* (AA * RR * BB')
    end
    R
end

function update_env_right(
    RE::S, A::S, M::T, B::S, ::Val{:c}
) where {T <: SparseSiteTensor, S <: AbstractArray{Float64, 3}}
    R = zeros(size(A, 1), maximum(M.projs[1]), size(B, 1))

    #Threads.@threads for σ ∈ 1:length(M.loc_exp)
    #    lexp = M.loc_exp[σ]
    for (σ, lexp) ∈ enumerate(M.loc_exp)
        AA = @view A[:, M.projs[4][σ], :]
        RR = @view RE[:, M.projs[3][σ], :]
        BB = @view B[:, M.projs[2][σ], :]
        R[:, M.projs[1][σ], :] += lexp .* (AA * RR * BB')
    end
    R
end

function update_env_right(
    RE::S, A::S, M::T, B::S, ::Val{:n}
) where {T <: SparseVirtualTensor, S <: AbstractArray{Float64,3}}
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs
    @cast A4[x, k, l, y] := A[x, (k, l), y] (k ∈ 1:maximum(p_rt))
    @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lb))

    R = zeros(size(A, 1), length(p_l), size(B, 1))
    for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
        AA = @view A4[:, p_rt[r], p_lt[l], :]
        RR = @view RE[:, r, :]
        BB = @view B4[:, p_lb[l], p_rb[r], :]
        R[:, l, :] += h[p_l[l], p_r[r]] * (AA * RR * BB')
    end
    R
end

function update_env_right(
    RE::S, A::S, M::T, B::S, ::Val{:c}
) where {T <: SparseVirtualTensor, S <: AbstractArray{Float64, 3}}
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs
    @cast A4[x, k, l, y] := A[x, (k, l), y] (k ∈ 1:maximum(p_rt))
    @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lb))

    R = zeros(size(A, 1), length(p_l), size(B, 1))
    for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
        AA = @view A4[:, p_lb[l], p_rb[r], :]
        RR = @view RE[:, r, :]
        BB = @view B4[:, p_rt[r], p_lt[l], :]
        R[:, l, :] += h[p_l[l], p_r[r]] * (AA * RR * BB')
    end
    R
end

function update_env_right(
    RE::S, A₀::S, M::T, B₀::S, trans::Symbol=:n
) where {T <: AbstractDict, S} # {T <: AbstractDict, S <: AbstractArray{Float64, 3}}
    sites = sort(collect(keys(M)))
    A = _update_tensor_forward(A₀, M, sites, Val(trans))
    B = _update_tensor_backwards(B₀, M, sites, Val(trans))
    update_env_right(RE, A, M[0], B, Val(trans))
end

#           --
#              |
# R =  -- M -- RE
#              |
#           --
function update_env_right(
    RE::S, M::T, ::Val{:c}
) where {S, T <: AbstractDict} # {S <: AbstractArray{Float64, 3}, T <: AbstractDict}
    MM = M[0]
    @tensor R[nt, nc, nb] := MM[nc, oc] * RE[nt, oc, nb]
    R
end

function update_env_right(
    RE::S, M::T, ::Val{:n}
) where {S, T <: AbstractDict} # {S <: AbstractArray{Float64, 3}, T <: AbstractDict}
    MM = M[0]
    @tensor R[nt, nc, nb] := MM[nc, oc] * RE[nt, oc, nb]
    R
end

function project_ket_on_bra(env::Environment, site::Site, trans::Symbol=:n)
    project_ket_on_bra(
        env.env[(site, :left)],
        env.ket[site],
        env.mpo[site],
        env.env[(site, :right)],
        Val(trans)
    )
end

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


#   |    |    |
#  LE -- M -- RE
#   |    |    |
#     -- B --
function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{:n}
) where {T <: AbstractArray{Float64, 4}, S <: AbstractArray{Float64, 3}}
    @tensor A[x, y, z] := LE[k, l, x] * B[k, m, o] *
                          M[l, y, n, m] * RE[z, n, o] order = (k, l, m, n, o)
    A
end

function project_ket_on_bra(
    LE::S, B::S, C::S, M::T, N::T, RE::S, ::Val{:n}
) where {T <: AbstractArray{Float64, 4}, S <: AbstractArray{Float64, 3}}
    @tensor A[x, y, z, r] := LE[k, l, x] * B[k, m, o] *
                          M[l, y, n, m] * C[o, s, q] *
                          N[n, z, p, s] * RE[r, p, q] order = (k, l, m, n, o, s, p, q)
    A
end

function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{:n}
) where {T <: AbstractArray{Float64, 2}, S <: AbstractArray{Float64, 3}}
    @tensor A[x, y, z] := B[x, a, z] * M[y, a]
    A
end

function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{:c}
) where {T <: AbstractArray{Float64, 2}, S <: AbstractArray{Float64, 3}}
    @tensor A[x, y, z] := B[x, a, z] * M[a, y]
    A
end

function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: SparseSiteTensor}
    A = zeros(size(LE, 3), maximum(M.projs[2]), size(RE, 1))

    #Threads.@threads for σ ∈ 1:length(M.loc_exp)
    #    lexp = M.loc_exp[σ]
    for (σ, lexp) ∈ enumerate(M.loc_exp)
        le = @view LE[:, M.projs[1][σ], :]
        b = @view B[:, M.projs[4][σ], :]
        re = @view RE[:, M.projs[3][σ], :]
        A[:, M.projs[2][σ], :] += lexp .* (le' * b * re')
    end
    A
end

function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}

    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs
    @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lb))
    A = zeros(size(LE, 3), maximum(p_rt), maximum(p_lt), size(RE, 1))
    for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
        le = @view LE[:, l, :]
        b = @view B4[:, p_lb[l], p_rb[r], :]
        re = @view RE[:, r, :]
        A[:,  p_rt[r], p_lt[l], :] += h[p_l[l], p_r[r]] .* (le' * b * re')
    end
    @cast AA[l, (ũ, u), r] := A[l, ũ, u, r]
    AA
end

function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{:c}
) where {T <: AbstractArray{Float64, 4}, S <: AbstractArray{Float64, 3}}
    @tensor A[x, y, z] := LE[k, l, x] * B[k, m, o] *
                          M[l, m, n, y] * RE[z, n, o] order = (k, l, m, n, o)
    A
end

function project_ket_on_bra(
    LE::S, B::S, C::S, M::T, N::T, RE::S, ::Val{:c}
) where {T <: AbstractArray{Float64, 4}, S <: AbstractArray{Float64, 3}}
    @tensor A[x, y, z, r] := LE[k, l, x] * B[k, m, o] *
                          M[l, y, n, m] * C[o, s, q] *
                          N[n, z, p, s] * RE[r, p, q] order = (k, l, m, n, o, s, p, q)
    A
end

function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: SparseSiteTensor}
    ## TO BE WRITTEN
    A = zeros(size(LE, 3), maximum(M.projs[4]), size(RE, 1))

    #Threads.@threads for σ ∈ 1:length(M.loc_exp)
    #    lexp = M.loc_exp[σ]
    for (σ, lexp) ∈ enumerate(M.loc_exp)
        le = @view LE[:, M.projs[1][σ], :]
        b = @view B[:, M.projs[2][σ], :]
        re = @view RE[:, M.projs[3][σ], :]
        A[:, M.projs[4][σ], :] += lexp .* (le' * b * re')
    end
    A
end

function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs
    @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lb))
    A = zeros(size(LE, 3), maximum(p_lb), maximum(p_rb), size(RE, 1))
    for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
        le = @view LE[:, l, :]
        b = @view B4[:, p_rt[r], p_lt[l], :]
        re = @view RE[:, r, :]
        A[:, p_lb[l], p_rb[r], :] += h[p_l[l], p_r[r]] .* (le' * b * re')
    end
    @cast AA[l, (ũ, u), r] := A[l, ũ, u, r]
    AA
end

function project_ket_on_bra(
    LE::S, B₀::S, M::T, RE::S, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: AbstractDict}
    sites = sort(collect(keys(M)))
    C = sort(collect(M), by = x -> x[1])
    TT = B₀
    for (t, v) ∈ reverse(C) TT = project_ket_on_bra(LE, TT, v, RE, Val(:n)) end
    TT
end

function project_ket_on_bra(
    LE::S, B₀::S, M::T, RE::S, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: AbstractDict}
    sites = sort(collect(keys(M)))
    C = sort(collect(M), by = x -> x[1])
    TT = B₀
    for (t, v) ∈ C TT = project_ket_on_bra(LE, TT, v, RE, Val(:c)) end
    TT
end

function measure_env(env::Environment, site::Site, trans::Symbol=:n)
    L = update_env_left(
        env.env[(site, :left)], env.bra[site], env.mpo[site], env.ket[site], trans
    )
    R = env.env[(site, :right)]
    @tensor L[t, c, b] * R[b, c, t]
end

function truncate!(ψ::QMPS, s::Symbol, Dcut::Int=typemax(Int), args...)
    @assert s ∈ (:left, :right)
    if s == :right
        _right_sweep!(ψ, args...)
        _left_sweep!(ψ, Dcut, args...)
    else
        _left_sweep!(ψ, args...)
        _right_sweep!(ψ, Dcut, args...)
    end
end
canonise!(ψ::QMPS, s::Symbol) = canonise!(ψ, Val(s))
canonise!(ψ::QMPS, ::Val{:right}) = _left_sweep!(ψ, typemax(Int))
canonise!(ψ::QMPS, ::Val{:left}) = _right_sweep!(ψ, typemax(Int))

function _right_sweep!(ψ::QMPS, Dcut::Int=typemax(Int), args...)
    R = ones(eltype(ψ[1]), 1, 1)
    for i ∈ ψ.sites
        A = ψ[i]
        @matmul M̃[(x, σ), y] := sum(α) R[x, α] * A[α, σ, y]
        Q, R = qr_fact(M̃, Dcut, args...)
        R = R ./ maximum(abs.(R))
        @cast A[x, σ, y] := Q[(x, σ), y] (σ ∈ 1:size(A, 2))
        ψ[i] = A
    end
end

function _left_sweep!(ψ::QMPS, Dcut::Int=typemax(Int), args...)
    R = ones(eltype(ψ[1]), 1, 1)
    for i ∈ reverse(ψ.sites)
        B = ψ[i]
        @matmul M̃[x, (σ, y)] := sum(α) B[x, σ, α] * R[α, y]
        R, Q = rq_fact(M̃, Dcut, args...)
        R = R ./ maximum(abs.(R))
        @cast B[x, σ, y] := Q[x, (σ, y)] (σ ∈ 1:size(B, 2))
        ψ[i] = B
    end
end
