export
    Environment,
    VariationalCompression,
    left_nbrs_site,
    right_nbrs_site

abstract type AbstractCompression end

mutable struct Environment{T} <: AbstractEnvironment
    bra::QMps{T}  # ψ to be optimized
    mpo::QMpo{T}
    ket::QMps{T}
    trans::Bool
    env::Dict

    function Environment(bra, mpo, ket; trans=false)
        @assert bra.sites == ket.sites && issubset(bra.sites, mpo.sites)
        T = promote_type(eltype.(bra, mpo, ket)...)
        id = ones(T, 1, 1, 1)
        env0 = Dict((first(bra.sites), :left) => id, (last(bra.sites), :right) => id)
        env = new{T}(bra, mpo, ket, trans, env0)
        for site ∈ env.bra.sites update_env_left!(env, site, trans=trans) end
        env
    end
end

struct VariationalCompression{T <: Real} <: AbstractCompression
    tol::T
    max_sweeps::Int
    trans::Bool
    args

    function VariationalCompression(; tol=1E-10, max_sweeps=4, trans=false, args)
        new{eltype(tol)}(tol, max_sweeps, trans, args)
    end
end

function (compress!::VariationalCompression)(bra::QMps, mpo::QMpo, ket::QMps)
    env = Environment(bra, mpo, ket, trans=compress.trans)
    overlap = Inf
    overlap_before = measure_env(env, last(env.bra.sites), compress.trans)

    for sweep ∈ 1:compress.max_sweeps
        _left_sweep_var!(env, compress.trans, compress.args...)
        _right_sweep_var!(env, compress.trans, compress.args...)

        overlap = measure_env(env, last(env.bra.sites), compress.trans)
        Δ = abs(overlap_before - abs(overlap))
        @info "Convergence" Δ

        if Δ < compress.tol
            @info "Finished in $sweep sweeps of $(max_sweeps)."
            return overlap
        else
            overlap_before = overlap
        end
    end
    overlap
end

function _left_sweep_var!(env::Environment; trans::Bool, args...)
    for site ∈ reverse(env.bra.sites)
        update_env_right!(env, site; trans=trans)
        A = project_ket_on_bra(env, site, trans=trans)
        @cast B[x, (y, z)] := A[x, y, z]
        _, Q = rq_fact(B, args...)
        @cast C[x, σ, y] := Q[x, (σ, y)] (σ ∈ 1:size(A, 2))
        env.bra[site] = C
        clear_env_containing_site!(env, site)
    end
end

function _right_sweep_var!(env::Environment; trans::Bool, args...)
    for site ∈ env.bra.sites
        update_env_left!(env, site; trans=trans)
        A = project_ket_on_bra(env, site, trans=trans)
        @cast B[(x, y), z] := A[x, y, z]
        Q, _ = qr_fact(B, args...)
        @cast C[x, σ, y] := Q[(x, σ), y] (σ ∈ 1:size(A, 2))
        env.bra[site] = C
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

function update_env_left!(env::Environment, site::Site; trans::Bool)
    site <= first(env.bra.sites) && return
    ls = left_nbrs_site(site, env.bra.sites)
    LL = update_env_left(env.env[(ls, :left)], env.bra[ls], env.mpo[ls], env.ket[ls], trans=trans)
    rs = right_nbrs_site(ls, env.mpo.sites)
    while rs < site
        LL = update_env_left(LL, env.mpo[rs]; trans=trans)
        rs = right_nbrs_site(rs, env.mpo.sites)
    end
    push!(env.env, (site, :left) => LL)
end

function update_env_right!(env::Environment, site::Site; trans::Bool)
    site >= last(env.bra.sites) && return
    rs = right_nbrs_site(site, env.bra.sites)
    RR = update_env_right(env.env[(rs, :right)], env.bra[rs], env.mpo[rs], env.ket[rs]; trans=trans)
    ls = left_nbrs_site(rs, env.mpo.sites)
    while ls > site
        RR = update_env_right(RR, env.mpo[ls]; trans=trans)
        ls = left_nbrs_site(ls, env.mpo.sites)
    end
    push!(env.env, (site, :right) => RR)
end

"""
        -- A --
      |    |
 L = LE -- M --
      |    |
        -- B --
"""
function update_env_left(LE::S, A₀::S, M::Dict, B₀::S; trans) where S <: ArrayOrCuArray{<:Real, 3}
    sites = sort(collect(keys(M)))
    A = _update_tensor_forward(A₀, M, sites, trans=trans)
    B = _update_tensor_backwards(B₀, M, sites, trans=trans)
    update_env_left(LE, A, M[0], B, trans=trans)
end

update_env_left(LE::ArrayOrCuArray{<:Real, 3}, M::Dict; trans) = attach_central_left(LE, M[0])

function _update_tensor(A::ArrayOrCuArray{<:Real, 3}, M::Dict, sites; trans::Bool, dir::Symbol)
    @assert dir ∈ (:forward, :backwards)
    B = copy(A)
    for i ∈ (trans ? reverse(sites) : sites)
        i == 0 && break
        B = (dir == :forward ? attach_central_left : attach_central_right)(B, M[i])
    end
    B
end

function update_env_right(RE::S, A₀::S, M::Dict, B₀::S; trans::Bool) where S <: ArrayOrCuArray{<:Real, 3}
    sites = sort(collect(keys(M)))
    A = _update_tensor_forward(A₀, M, sites; trans=trans)
    B = _update_tensor_backwards(B₀, M, sites; trans=trans)
    update_env_right(RE, A, M[0], B; trans=trans)
end

"""
           --
              |
 R =  -- M -- RE
              |
           --
"""
update_env_right(RE::ArrayOrCuArray{<:Real, 3}, M::Dict; trans::Bool) = attach_central_right(RE, M[0])

function project_ket_on_bra(env::Environment, s::Site; trans::Bool)
    project_ket_on_bra(env.env[(s, :left)], env.ket[s],  env.mpo[s], env.env[(s, :right)], trans)
end

function project_ket_on_bra(LE::S, B₀::S, M::Dict, RE::S, trans::Bool) where S <: ArrayOrCuArray{<:Real, 3}
    _attach = trans ? attach_central_left : attach_central_right
    C = sort(collect(M), by = x -> x[1])
    TT = B₀
    for (_, v) ∈ (trans ? C : reverse(C))
        TT = length(size(v)) == 2 ? _attach(TT, v) : project_ket_on_bra(LE, TT, v, RE, trans)
    end
    TT
end

function measure_env(env::Environment, s::Site, trans::Bool)
    L = update_env_left(env.env[(s, :left)], env.bra[s], env.mpo[s], env.ket[s], trans=trans)
    R = env.env[(s, :right)]
    @tensor L[t, c, b] * R[b, c, t]
end

function clear_env_containing_site!(env::Environment, s::Site)
    delete!(env.env, (left_nbrs_site(s, env.ket.sites), :right))
    delete!(env.env, (right_nbrs_site(s, env.ket.sites), :left))
end
