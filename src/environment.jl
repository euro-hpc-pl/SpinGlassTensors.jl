export
    Environment,
    left_nbrs_site,
    right_nbrs_site

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
        update_env_left!.(Ref(env), env.bra.sites)
        env
    end
end

function clear_env_containing_site!(env::Environment, site::Site)
    delete!(env.env, (left_nbrs_site(site, env.ket.sites), :right))
    delete!(env.env, (right_nbrs_site(site, env.ket.sites), :left))
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

"""
        -- A --
      |    |
 L = LE -- M --
      |    |
        -- B --
"""
function update_env_left(
    LE::S, A::S, M::T, B::S
) where {S <: CuArrayOrArray{R, 3}, T <: MpoTensor{R, 4}} where R <: Real
    for v ∈ M.top A = contract_tensor3_matrix(A, v) end
    for v ∈ reverse(M.bot) B = contract_matrix_tensor3(v, B) end
    update_env_left(LE, A, M.ctr, B)
end

function update_env_left!(env::Environment, site::Site)
    site <= first(env.bra.sites) && return
    ls = left_nbrs_site(site, env.bra.sites)
    LL = update_env_left(env.env[(ls, :left)], env.bra[ls], env.mpo[ls], env.ket[ls])
    rs = right_nbrs_site(ls, env.mpo.sites)
    while rs < site
        LL = contract_tensor3_matrix(LL, env.mpo[rs])
        rs = right_nbrs_site(rs, env.mpo.sites)
    end
    push!(env.env, (site, :left) => LL)
end

"""
      -- A --
         |    |
 R =  -- M -- RE
         |    |
      -- B --
"""
function update_env_right(
    RE::S, A::S1, M::T, B::S
) where {T <: MpoTensor{R, 4}, S <: CuArrayOrArray{R, 3}, S1 <: CuArrayOrArray{R, 3}} where R <: Real
    for v ∈ M.top  A = contract_tensor3_matrix(A, v) end
    for v ∈ reverse(M.bot) B = contract_matrix_tensor3(v, B) end
    update_env_right(RE, A, M.ctr, B)
end

function update_env_right!(env::Environment, site::Site)
    site >= last(env.bra.sites) && return
    rs = right_nbrs_site(site, env.bra.sites)
    RR = update_env_right(env.env[(rs, :right)], env.bra[rs], env.mpo[rs], env.ket[rs])
    ls = left_nbrs_site(rs, env.mpo.sites)
    while ls > site
        RR = contract_matrix_tensor3(env.mpo[ls], RR)
        ls = left_nbrs_site(ls, env.mpo.sites)
    end
    push!(env.env, (site, :right) => RR)
end

"""
   |    |    |
  LE -- M -- RE
   |    |    |
     -- B --
"""
function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S
) where {S <: CuArrayOrArray{R, 3}, T <: MpoTensor{R, 4}} where R <: Real
    for v ∈ reverse(M.bot) B = contract_matrix_tensor3(v, B) end
    B = project_ket_on_bra(LE, B, M.ctr, RE)
    for v ∈ reverse(M.top) B = contract_matrix_tensor3(v, B) end
    B
end

function project_ket_on_bra(env::Environment, site::Site)
    project_ket_on_bra(
        env.env[(site, :left)], env.ket[site], env.mpo[site], env.env[(site, :right)]
    )
end

function measure_env(env::Environment, site::Site)
    L = update_env_left(env.env[(site, :left)], env.bra[site], env.mpo[site], env.ket[site])
    R = env.env[(site, :right)]
    @tensor L[b, c, t] * R[t, c, b]
end