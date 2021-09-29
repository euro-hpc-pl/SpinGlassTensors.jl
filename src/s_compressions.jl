mutable struct Environment <: AbstractEnvironment
    bra::Mps  # to be optimized
    mpo::Mpo
    ket::Mps
    trans::Symbol
    env::Dict

    function Environment(
        bra::Mps,
        mpo::Mpo,
        ket::Mps,
        trans::Symbol=:c
    )
        @assert trans ∈ (:n, :c)
        @assert bra.sites == ket.sites
        @assert issubset(bra.sites, mpo.sites)

        env = Dict(
                   (first(bra.sites), :left) => ones(1, 1, 1),
                   (last(bra.sites), :right) => ones(1, 1, 1)
            )
        environment = new(bra, mpo, ket, trans, env)
        for site ∈ environment.bra.sites 
            update_env_left!(environment, site) 
        end
        environment
    end
end


function SpinGlassTensors.compress!(
    bra::Mps,
    mpo::Mpo,
    ket::Mps,
    Dcut::Int,
    tol::Number=1E-8,
    max_sweeps::Int=4,
    args...
)
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
    overlap
end


function _left_sweep_var!(env::Environment, args...)
    for site ∈ reverse(env.bra.sites)
        update_env_right!(env, site)
        A = project_ket_on_bra(env, site)
        @cast B[x, (y, z)] := A[x, y, z]
        _, Q = rq_fact(B, args...)
        @cast C[x, σ, y] := Q[x, (σ, y)] (σ ∈ 1:size(A, 2))
        env.bra[site] = C
        clear_env_site!(env, site) 
    end
end


function _right_sweep_var!(env::Environment, args...)
    for site ∈ env.bra.sites
        update_env_left!(env, site)
        A = project_ket_on_bra(env, site)
        @cast B[(x, y), z] := A[x, y, z]
        Q, _ = qr_fact(B, args...)
        @cast C[x, σ, y] := Q[(x, σ), y] (σ ∈ 1:size(A, 2))
        env.bra[site] = C
        clear_env_site!(env, site)
    end
end


function _left_nbrs_site(site::Site, sites::NTuple{N, Site}) where N
    # largest x in sites: x < site
    ls = filter(i -> i < site, sites)
    if isempty(ls) return -Inf end
    maximum(ls)
end


function _right_nbrs_site(site::Site, sites::NTuple{N, Site}) where N
    # smallest x in sites: x > site
    ms = filter(i -> i > site, sites)
    if isempty(ms) return Inf end
    minimum(ms)
end


function update_env_left!(env::Environment, site::Site)
    if site <= first(env.bra.sites) return end

    ls = _left_nbrs_site(site, env.bra.sites)
    LL = update_env_left!(
            env.env[(ls, :left)],
            env.bra[ls],
            env.mpo[ls],
            env.ket[ls]
    )

    rs = _right_nbrs_site(ls, env.mpo.sites)

    while rs < site
        M = env.mpo[rs]
        LL = update_env_left(LL, M)
        rs = _right_nbrs_site(rs, env.mpo.sites)
    end
    push!(env.env, (site, :left) => LL)
end


function update_env_right!(env::Environment, site::Site)
    if site >= last(env.bra.sites) return end

    rs = _right_nbrs_site(site, env.bra.sites)
    RR = update_env_right!(  
            env.env[(rs, :right)],
            env.bra[rs],
            env.mpo[rs],
            env.ket[rs]
    )

    ls = _left_nbrs_site(rs, env.mpo.sites)
    while ls > site
        M = env.mpo[ls]
        RR = update_env_right(RR, M)
        ls = _left_nbrs_site(ls, env.mpo.sites)
    end
    push!(env.env, (site, :right) => RR)
end


function clear_env_site!(env::Environment, site::Site)
    # delete!(env.env, (site, :right))
    # delete!(env.env, (site, :left))
end


#        -- A --
#      |    |
# L = LE -- M --  
#      |    |
#        -- B --
#
function update_env_left(LE::S, A::S, M::T, B::S) where {S, T <: AbstractArray} 
    # for real there is no conjugate, otherwise conj(A)
    @tensor L[nb, nc, nt] := LE[ob, oc, ot] * A[ot, α, nt] * 
                             M[oc, α, nc, β] * B[ob, β, nb] order = (ot, α, oc, β, ob)  
    L
end


function update_env_left(LE::S, A::S, M::T, B::S, ::Val{:c}) where {S, T <: AbstractArray} 
    # for real there is no conjugate, otherwise conj(A)
    @tensor L[nb, nc, nt] := LE[ob, oc, ot] * A[ot, α, nt] * 
                             M[oc, β, nc, α] * B[ob, β, nb] order = (ot, α, oc, β, ob)  
    L
end


function _update_tensor_forward!(A::T, M::Dict, sites::NTuple{N, Site}) where {N, T <: AbstractArray} 
    for i ∈ sites
        if i == 0 return end
        B = M[i]
        @tensor A[l, x, r] := A[l, y, r] * B[y, x]
    end
end


function _update_tensor_backwards!(A::T, M::Dict, sites::NTuple{N, Site}) where {N, T <: AbstractArray} 
    for i ∈ reverse(sites)
        if i == 0 return end
        B = M[i]
        @tensor A[l, x, r] := A[l, y, r] * B[x, y]
    end
end


function update_env_left!(LE::T, A₀::S, M::Dict, B₀::S) where {T, S <: AbstractArray} 
    sites = collect(sort(keys(M)))
    _update_tensor_forward!(A₀, M, sites)
    _update_tensor_backwards!(B₀, M, sites)
    update_env_left(LE, A₀, M[0], B₀)
end


#      -- A --
#         |    |
# R =  -- M -- RE 
#         |    |
#      -- B --
#
function update_env_right(RE::S, A::S, M::T, B::S) where {T, S <: AbstractArray}
    # for real there is no conjugate, otherwise conj(A)
    @tensor R[nt, nc, nb] := RE[ot, oc, ob] * A[nt, α, ot] *
                             M[nc, α, oc, β] * B[nb, β, ob] order = (ot, α, oc, β, ob)
    R
end


function update_env_right(RE::S, A::S, M::T, B::S, ::Val{:c}) where {T, S <: AbstractArray}
    # for real there is no conjugate, otherwise conj(A)
    @tensor R[nt, nc, nb] := RE[ot, oc, ob] * A[nt, α, ot] * 
                             M[nc, β, oc, α] * B[nb, β, ob] order = (ot, α, oc, β, ob)
    R
end


function update_env_right!(RE::S, A₀::S, M::Dict, B₀::S) where {T, S <: AbstractArray} 
    sites = collect(sort(keys(M)))
    _update_tensor_forward!(A₀, M, sites)
    _update_tensor_backwards!(B₀, M, sites)
    update_env_right(RE, A₀, M[0], B₀)
end


#           --
#              |
# R =  -- M -- RE 
#              |
#           --
#
function update_env_right(RE::T, M::S) where {T, S <: AbstractArray}  
    @tensor R[nt, nc, nb] := M[nc, oc] * RE[nt, oc, nb]
    R
end



function project_ket_on_bra(env::Environment, site::Site)
    project_ket_on_bra(
        env.env[(site, :left)],
        env.ket[site],
        env.mpo[site],
        env.env[(site, :right)]
    )
end


#   |    |    |
#  LE -- M -- RE 
#   |    |    |
#     -- B --
#
function project_ket_on_bra(LE::S, B::S, M::T, RE::S) where {T, S <: AbstractArray}
    @tensor A[x, y, z] := LE[k, l, x] * B[k, m, o] * 
                          M[l, y, n, m] * RE[z, n, o] order = (k, l, m, n, o)
    A
end


function project_ket_on_bra(LE::S, B::S, M::T, RE::S, ::Val{:c}) where {T, S <: AbstractArray}
    @tensor A[x, y, z] := LE[k, l, x] * B[k, m, o] * 
                          M[l, m, n, y] * RE[z, n, o] order = (k, l, m, n, o)
    A
end


function project_ket_on_bra(LE::S, B₀::S, M::Dict, RE::S) where S <: AbstractArray
    sites = collect(sort(keys(M)))
    _update_tensor_forward!(B₀, M, sites)
    T = project_ket_on_bra(LE, B₀, M[0], RE)
    _update_tensor_backwards!(T, M, sites)
    T
end


function measure_env(env::Environment, site::Site)
    L = update_env_left(
            env.env[(site, :left)],
            env.bra[site],
            env.mpo[site],
            env.ket[site],
    )
    R = env.env[(site, :right)]
    @tensor L[t, c, b] * R[b, c, t]
end


function truncate!(ψ::Mps, s::Symbol, Dcut::Int=typemax(Int), args...)
    @assert s ∈ (:left, :right)
    if s == :right
        _right_sweep!(ψ, args...)
        _left_sweep!(ψ, Dcut, args...)
    else
        _left_sweep!(ψ, args...)
        _right_sweep!(ψ, Dcut, args...)
    end
end


canonise!(ψ::Mps, s::Symbol) = canonise!(ψ, Val(s))
canonise!(ψ::Mps, ::Val{:right}) = _left_sweep!(ψ, typemax(Int))
canonise!(ψ::Mps, ::Val{:left}) = _right_sweep!(ψ, typemax(Int))


function _right_sweep!(ψ::Mps, Dcut::Int=typemax(Int), args...)
    R = ones(eltype(ψ.tensors[1]), 1, 1)
    for i ∈ ψ.sites      
        A = ψ.tensors[i]
        @matmul M̃[(x, σ), y] := sum(α) R[x, α] * A[α, σ, y]
        Q, R = qr_fact(M̃, Dcut, args...)
        R = R ./ maximum(abs.(R))
        @cast A[x, σ, y] := Q[(x, σ), y] (σ ∈ 1:size(A, 2))
        ψ.tensors[i] = A
    end
end


function _left_sweep!(ψ::Mps, Dcut::Int=typemax(Int), args...)
    R = ones(eltype(ψ.tensors[1]), 1, 1)
    for i ∈ length(ψ.sites):-1:1
        B = ψ.tensors[i]   
        @matmul M̃[x, (σ, y)] := sum(α) B[x, σ, α] * R[α, y]
        R, Q = rq_fact(M̃, Dcut, args...)
        R = R ./ maximum(abs.(R))
        @cast B[x, σ, y] := Q[x, (σ, y)] (σ ∈ 1:size(B, 2))
        ψ.tensors[i] = B
    end
end