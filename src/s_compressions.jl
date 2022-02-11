export compress!, compress_twosite!

"""
$(TYPEDEF)

$(TYPEDFIELDS)

"""
mutable struct Environment{T <: Real, S}
    bra::QMPS{T}
    mpo::QMPO{T}
    ket::QMPS{T}
    env::Dict{Tuple{Site, Symbol}, Tensor{T}}

    function Environment{T, S}(
        bra::QMPS{T},
        mpo::QMPO{T},
        ket::QMPS{T},
    ) where {T <: Real, S}
        @assert S ∈ (:n, :c)
        @assert bra.sites == ket.sites
        @assert issubset(bra.sites, mpo.sites)

        env0 = Dict(
            (first(bra.sites), :left) => ones(1, 1, 1),
            (last(bra.sites), :right) => ones(1, 1, 1)
        )
        env = new(bra, mpo, ket, env0)
        for site ∈ env.bra.sites update_env_left!(env, site) end
        env
    end
end

"""
$(TYPEDSIGNATURES)

"""
function Environment(bra::QMPS, mpo::QMPO, ket::QMPS, trans::Symbol=:n)
    T = reduce(promote_type, eltype.((bra, mpo, ket)))
    Environment{T, trans}(bra, mpo, ket)
end

"""
$(TYPEDSIGNATURES)

"""
trans(::Environment{T, S}) where {T, S} = S

"""
$(TYPEDSIGNATURES)

"""
function SpinGlassTensors.compress!(
    bra::QMPS,
    mpo::QMPO,
    ket::QMPS;
    tol::Real=1E-8,
    max_sweeps::Int=4,
)
    env = Environment(bra, mpo, ket)
    overlap = Inf
    overlap_before = measure_env(env, last(env.bra.sites))

    for sweep ∈ 1:max_sweeps
        _left_sweep_var!(env)
        _right_sweep_var!(env)

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

"""
$(TYPEDSIGNATURES)

"""
function compress_twosite!(
    bra::QMPS, mpo::QMPO, ket::QMPS, Dcut::Int, tol::Real=1E-8, max_sweeps::Int=4
)
    env = Environment(bra, mpo, ket)
    overlap = Inf
    overlap_before = measure_env(env, last(env.bra.sites))
    for sweep ∈ 1:max_sweeps
        _left_sweep_var_twosite!(env, Dcut, tol)
        _right_sweep_var_twosite!(env, Dcut, tol)

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

"""
$(TYPEDSIGNATURES)

"""
function _left_sweep_var!(env::Environment)
    for site ∈ reverse(env.bra.sites)
        update_env_right!(env, site)
        A = project_ket_on_bra(env, site)
        @cast B[x, (y, z)] := A[x, y, z]
        _, Q = rq_fact(B)
        @cast C[x, σ, y] := Q[x, (σ, y)] (σ ∈ 1:size(A, 2))
        env.bra[site] = C
        clear_env_containing_site!(env, site)
    end
end

"""
$(TYPEDSIGNATURES)

"""
function _right_sweep_var!(env::Environment)
    for site ∈ env.bra.sites
        update_env_left!(env, site)
        A = project_ket_on_bra(env, site)
        @cast B[(x, y), z] := A[x, y, z]
        Q, _ = qr_fact(B)
        @cast C[x, σ, y] := Q[(x, σ), y] (σ ∈ 1:size(A, 2))
        env.bra[site] = C
        clear_env_containing_site!(env, site)
    end
end

"""
$(TYPEDSIGNATURES)

"""
function _left_sweep_var_twosite!(env::Environment, Dcut::Int, tol::Number)
    for site ∈ reverse(env.bra.sites[2:end])
        update_env_right!(env, site)
        A = project_ket_on_bra_twosite(env, site)
        @cast B[(x, y), (z, w)] := A[x, y, z, w]
        U, S, VV = svd(B, Dcut, tol)
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

"""
$(TYPEDSIGNATURES)

"""
function _right_sweep_var_twosite!(env::Environment, Dcut::Int, tol::Number)
    for site ∈ env.bra.sites[1:end-1]
        site_r = _right_nbrs_site(site, env.bra.sites)
        update_env_left!(env, site)
        A = project_ket_on_bra_twosite(env, site_r)
        @cast B[(x, y), (z, w)] := A[x, y, z, w]
        U, S, V = svd(B, Dcut, tol)
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
"""
$(TYPEDSIGNATURES)

"""
function _left_nbrs_site(site::Site, sorted_sites)
    site_pos = searchsortedfirst(sorted_sites, site)
    site_pos == 1 ? -Inf : sorted_sites[site_pos-1]
end

# Smallest x in sites: x > site
"""
$(TYPEDSIGNATURES)

"""
function _right_nbrs_site(site::Site, sorted_sites)
    site_pos = searchsortedlast(sorted_sites, site)
    site_pos == length(sorted_sites) ? Inf : sorted_sites[site_pos+1]
end

"""
$(TYPEDSIGNATURES)

"""
function update_env_left!(env::Environment, site::Site)
    if site <= first(env.bra.sites) return end

    ls = _left_nbrs_site(site, env.bra.sites)
    LL = update_env_left(env.env[(ls, :left)], env.bra[ls], env.mpo[ls], env.ket[ls], trans(env))

    rs = _right_nbrs_site(ls, env.mpo.sites)
    while rs < site
        @tensor LL[nt, nc, nb] :=  LE[nt, oc, nb] * env.mpo[rs][0][oc, nc]
        rs = _right_nbrs_site(rs, env.mpo.sites)
    end
    push!(env.env, (site, :left) => LL)
end

"""
$(TYPEDSIGNATURES)

"""
function update_env_right!(env::Environment, site::Site)
    if site >= last(env.bra.sites) return end

    rs = _right_nbrs_site(site, env.bra.sites)
    RR = update_env_right(env.env[(rs, :right)], env.bra[rs], env.mpo[rs], env.ket[rs], trans(env))

    ls = _left_nbrs_site(rs, env.mpo.sites)
    while ls > site
        @tensor RR[nt, nc, nb] := env.mpo[ls][0][nc, oc] * RE[nt, oc, nb]
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

#        -- A --
#      |    |
# L = LE -- M --
#      |    |
#        -- B --
"""
$(TYPEDSIGNATURES)

"""
function update_env_left(
    LE::AbstractArray{T, 3},
    A₀::AbstractArray{T, 3},
    M::Dict,
    B₀::AbstractArray{T, 3},
    trans::Symbol
) where {T <: Real}
    sites = sort(collect(keys(M)))
    A =_update_tensor_forward(A₀, M, sites, Val(trans))
    B = _update_tensor_backwards(B₀, M, sites, Val(trans))

    if trans == :c
        @tensor L[nb, nc, nt] := LE[ob, oc, ot] * A[ot, α, nt] *
                                 M[0][oc, α, nc, β] * B[ob, β, nb] order = (ot, α, oc, β, ob)
    else
        @tensor L[nb, nc, nt] := LE[ob, oc, ot] * A[ot, α, nt] *
                                 M[0][oc, β, nc, α] * B[ob, β, nb] order = (ot, α, oc, β, ob)
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
        @tensor B[l, x, r] := B[l, y, r] * C[y, x]
    end
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
        @tensor B[l, x, r] := B[l, y, r] * C[x, y]
    end
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
        @tensor B[l, x, r] := B[l, y, r] * C[x, y]
    end
    B
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
        @tensor B[l, x, r] := B[l, y, r] * C[y, x]
    end
    B
end

"""
$(TYPEDSIGNATURES)

"""
function update_env_right(
    RE::S, A₀::S, M::T, B₀::S, trans::Symbol=:n
) where {T <: Dict, S} # {T <: AbstractDict, S <: AbstractArray{Float64, 3}}
    sites = sort(collect(keys(M)))
    A = _update_tensor_forward(A₀, M, sites, Val(trans))
    B = _update_tensor_backwards(B₀, M, sites, Val(trans))
    if trans == :n
        @tensor RR[nt, nc, nb] := RE[ot, oc, ob] * A[nt, α, ot] *
                                  M[0][nc, β, oc, α] * B[nb, β, ob] order = (ot, α, oc, β, ob)
    else
        @tensor RR[nt, nc, nb] := RE[ot, oc, ob] * A[nt, α, ot] *
                                  M[0][nc, α, oc, β] * B[nb, β, ob] order = (ot, α, oc, β, ob)
    end
    RR
end

"""
$(TYPEDSIGNATURES)

"""
function project_ket_on_bra(env::Environment, site::Site)
    LE = env.env[(site, :left)]
    B₀ = env.ket[site]
    M = env.mpo[site]
    RE = env.env[(site, :right)]

    sites = sort(collect(keys(M)), rev=trans(env)==:n)
    TT = B₀
    for k ∈ sites TT = project_ket_on_bra(LE, TT, M[k], RE, Val(trans(env))) end
    TT
end


#   |    |    |
#  LE -- M -- RE
#   |    |    |
#     -- B --
"""
$(TYPEDSIGNATURES)

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
    ::S, B::S, M::T, ::S, ::Val{:n}
) where {T <: AbstractArray{Float64, 2}, S <: AbstractArray{Float64, 3}}
    @tensor A[x, y, z] := B[x, a, z] * M[y, a]
    A
end

"""
$(TYPEDSIGNATURES)

"""
function project_ket_on_bra(
    ::S, B::S, M::T, ::S, ::Val{:c}
) where {T <: AbstractArray{Float64, 2}, S <: AbstractArray{Float64, 3}}
    @tensor A[x, y, z] := B[x, a, z] * M[a, y]
    A
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
function project_ket_on_bra_twosite(env::Environment, site::Site)
    site_l = _left_nbrs_site(site, env.bra.sites)
    LE = env.env[(site_l, :left)]
    B = env.ket[site_l]
    C = env.ket[site]
    M = env.mpo[site_l][0]
    N = env.mpo[site][0]
    RE = env.env[(site, :right)]

    @tensor A[x, y, z, r] := LE[k, l, x] * B[k, m, o] *
                          M[l, y, n, m] * C[o, s, q] *
                          N[n, z, p, s] * RE[r, p, q] order = (k, l, m, n, o, s, p, q)
    A
end

"""
$(TYPEDSIGNATURES)

"""
function measure_env(env::Environment, site::Site)
    L = update_env_left(env.env[(site, :left)], env.bra[site], env.mpo[site], env.ket[site], trans(env))
    R = env.env[(site, :right)]
    @tensor L[t, c, b] * R[b, c, t]
end

"""
$(TYPEDSIGNATURES)

"""
function truncate!(ψ::QMPS, s::Symbol, Dcut::Int=typemax(Int))
    @assert s ∈ (:left, :right)
    if s == :right
        _right_sweep!(ψ)
        _left_sweep!(ψ, Dcut)
    else
        _left_sweep!(ψ)
        _right_sweep!(ψ, Dcut)
    end
end

"""
$(TYPEDSIGNATURES)

"""
canonise!(ψ::QMPS, s::Symbol) = canonise!(ψ, Val(s))

"""
$(TYPEDSIGNATURES)

"""
canonise!(ψ::QMPS, ::Val{:right}) = _left_sweep!(ψ, typemax(Int))

"""
$(TYPEDSIGNATURES)

"""
canonise!(ψ::QMPS, ::Val{:left}) = _right_sweep!(ψ, typemax(Int))

"""
$(TYPEDSIGNATURES)

"""
function _right_sweep!(ψ::QMPS, Dcut::Int=typemax(Int))
    R = ones(eltype(ψ[1]), 1, 1)
    for i ∈ ψ.sites
        A = ψ[i]
        @matmul M̃[(x, σ), y] := sum(α) R[x, α] * A[α, σ, y]
        Q, R = qr_fact(M̃, Dcut)
        R = R ./ maximum(abs.(R))
        @cast A[x, σ, y] := Q[(x, σ), y] (σ ∈ 1:size(A, 2))
        ψ[i] = A
    end
end

"""
$(TYPEDSIGNATURES)

"""
function _left_sweep!(ψ::QMPS, Dcut::Int=typemax(Int))
    R = ones(eltype(ψ[1]), 1, 1)
    for i ∈ reverse(ψ.sites)
        B = ψ[i]
        @matmul M̃[x, (σ, y)] := sum(α) B[x, σ, α] * R[α, y]
        R, Q = rq_fact(M̃, Dcut)
        R = R ./ maximum(abs.(R))
        @cast B[x, σ, y] := Q[x, (σ, y)] (σ ∈ 1:size(B, 2))
        ψ[i] = B
    end
end
