export
    variational_compress!,
    left_nbrs_site,
    right_nbrs_site,
    variational_sweep!,
    Environment

abstract type AbstractEnvironment end

mutable struct Environment{T <: Real} <: AbstractEnvironment
    bra::QMps{T}  # ψ to be optimized
    mpo::QMpo{T}
    ket::QMps{T}
    env::Dict

    function Environment(bra::QMps{T}, mpo::QMpo{T}, ket::QMps{T}) where T <: Real
        @assert bra.sites == ket.sites && issubset(bra.sites, mpo.sites)
        id = ones(T, 1, 1, 1)
        env0 = Dict((bra.sites[1], :left) => id, (bra.sites[end], :right) => id)
        env = new{T}(bra, mpo, ket, env0)
        for site ∈ env.bra.sites update_env_left!(env, site) end
        env
    end
end

function variational_compress!(
    bra::QMps{T}, mpo::QMpo{T}, ket::QMps{T}, tol::Real=1E-10, max_sweeps::Int=4, args...
) where T <: Real
    env = Environment(bra, mpo, ket)
    overlap = Inf
    overlap_before = measure_env(env, last(env.bra.sites))

    for sweep ∈ 1:max_sweeps
        _left_sweep_var!(env, args...)
        _right_sweep_var!(env, args...)

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
    overlap, env
end

function _left_sweep_var!(env::Environment, args...)
    for site ∈ reverse(env.bra.sites)
        update_env_right!(env, site)
        A = project_ket_on_bra(env, site)
        @cast B[x, (y, z)] := A[x, y, z]
        _, Q = rq_fact(B, args...)
        @cast C[x, σ, y] := Q[x, (σ, y)] (σ ∈ 1:size(A, 2))
        env.bra[site] = Array(C)
        clear_env_containing_site!(env, site)
    end
end

function _right_sweep_var!(env::Environment, args...)
    for site ∈ env.bra.sites
        update_env_left!(env, site)
        A = project_ket_on_bra(env, site)
        @cast B[(x, y), z] := A[x, y, z]
        Q, _ = qr_fact(B, args...)
        @cast C[x, σ, y] := Q[(x, σ), y] (σ ∈ 1:size(A, 2))
        env.bra[site] = Array(C)
        clear_env_containing_site!(env, site)
    end
end

"""
Largest x in sites: x < site
"""
function left_nbrs_site(site::Site, sites)
    ls = filter(i -> i < site, sites)
    isempty(ls) && return -Inf
    maximum(ls)
end

"""
Smallest x in sites: x > site
"""
function right_nbrs_site(site::Site, sites)
    ms = filter(i -> i > site, sites)
    isempty(ms) && return Inf
    minimum(ms)
end

function update_env_left!(env::Environment, site::Site)
    site <= first(env.bra.sites) && return

    ls = left_nbrs_site(site, env.bra.sites)
    LL = update_env_left(env.env[(ls, :left)], env.bra[ls], env.mpo[ls], env.ket[ls])

    rs = right_nbrs_site(ls, env.mpo.sites)
    while rs < site
        LL = update_env_left(LL, env.mpo[rs])
        rs = right_nbrs_site(rs, env.mpo.sites)
    end
    push!(env.env, (site, :left) => LL)
end

function update_env_right!(env::Environment, site::Site)
    site >= last(env.bra.sites) && return

    rs = right_nbrs_site(site, env.bra.sites)
    RR = update_env_right(env.env[(rs, :right)], env.bra[rs], env.mpo[rs], env.ket[rs])

    ls = left_nbrs_site(rs, env.mpo.sites)
    while ls > site
        RR = update_env_right(RR, env.mpo[ls])
        ls = left_nbrs_site(ls, env.mpo.sites)
    end
    push!(env.env, (site, :right) => RR)
end

function clear_env_containing_site!(env::Environment, site::Site)
    delete!(env.env, (left_nbrs_site(site, env.ket.sites), :right))
    delete!(env.env, (right_nbrs_site(site, env.ket.sites), :left))
end

"""
        -- A --
      |    |
 L = LE -- M --
      |    |
        -- B --
"""
function update_env_left(
    LE::S, A₀::S, M::T, B₀::S
) where {S <: CuArrayOrArray{R, 3}, T <: TensorMap{R}} where R <: Real
    sites = sort(collect(keys(M)))
    A = _update_tensor_forward(A₀, M, sites)
    B = _update_tensor_backwards(B₀, M, sites)
    update_env_left(LE, A, M[0], B)
end

function update_env_left(
    LE::S, M::T
) where {S <: CuArrayOrArray{R, 3}, T <: TensorMap{R}} where R <: Real
    attach_central_left(LE, M[0])
end

projector_to_dense(::Type{T}, pr::Array{Int, 1}) where T = diagm(ones(T, maximum(pr)))[:, pr]
projector_to_dense(pr::Array{Int, 1}) = projector_to_dense(Float64, pr) # TODO to be removed

function _update_tensor_forward(
    A::S, M::T, sites
) where {S <: CuArrayOrArray{R, 3}, T <: TensorMap{R}} where R <: Real
    B = copy(A)
    for i ∈ sites
        i == 0 && break
        B = attach_central_left(B, M[i])
    end
    B
end

function _update_tensor_backwards(
    A::S, M::T, sites
) where {S <: CuArrayOrArray{R, 3}, T <: TensorMap{R}} where R <: Real
    B = copy(A)
    for i ∈ reverse(sites)
        i == 0 && break
        B = attach_central_right(B, M[i])
    end
    B
end

function update_env_right(
    RE::S, A₀::S1, M::T, B₀::S
) where {T <: TensorMap{R}, S <: CuArrayOrArray{R, 3}, S1 <: CuArrayOrArray{R, 3}} where R <: Real
    sites = sort(collect(keys(M)))
    A = _update_tensor_forward(A₀, M, sites)
    B = _update_tensor_backwards(B₀, M, sites)
    update_env_right(RE, A, M[0], B)
end

"""
           --
              |
 R =  -- M -- RE
              |
           --
"""
function update_env_right(
    RE::S, M::T
) where {S <: CuArrayOrArray{R, 3}, T <: TensorMap{R}} where R <: Real
    attach_central_right(RE, M[0])
end

function project_ket_on_bra(env::Environment, site::Site)
    project_ket_on_bra(
        env.env[(site, :left)],
        env.ket[site],
        env.mpo[site],
        env.env[(site, :right)]
    )
end

function project_ket_on_bra(
    LE::S, B₀::S, M::T, RE::S
) where {S <: CuArrayOrArray{R, 3}, T <: TensorMap{R}} where R <: Real
    C = sort(collect(M), by = x -> x[1])
    TT = B₀
    for (_, v) ∈ reverse(C)
        dimv = length(size(v))
        if dimv == 2
            TT = attach_central_right(TT, v)
        else
            TT = project_ket_on_bra(LE, TT, v, RE)
        end
    end
    TT
end

function measure_env(env::Environment, site::Site)
    L = update_env_left(env.env[(site, :left)], env.bra[site], env.mpo[site], env.ket[site])
    R = env.env[(site, :right)]
    @tensor L[t, c, b] * R[b, c, t]
end

function variational_sweep!(
    bra::QMps{T}, mpo::QMpo{T}, ket::QMps{T}, ::Val{:left}, args...
) where T <: Real
    env = Environment(bra, mpo, ket)
    _right_sweep_var!(env, args...)
end

function variational_sweep!(
    bra::QMps{T}, mpo::QMpo{T}, ket::QMps{T}, ::Val{:right}, args...
) where T <: Real
    env = Environment(bra, mpo, ket)
    _left_sweep_var!(env, args...)
end
