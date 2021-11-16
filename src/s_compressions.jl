export compress!
export _left_nbrs_site
export _right_nbrs_site


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
        for site ∈ environment.bra.sites update_env_left!(environment, site) end
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
        env.bra[site] = C
        clear_env_containing_site!(env, site)
    end
end


function _left_nbrs_site(site::Site, sites)
    # largest x in sites: x < site
    ls = filter(i -> i < site, sites)
    if isempty(ls) return -Inf end
    maximum(ls)
end


function _right_nbrs_site(site::Site, sites)
    # smallest x in sites: x > site
    ms = filter(i -> i > site, sites)
    if isempty(ms) return Inf end
    minimum(ms)
end


function update_env_left!(env::Environment, site::Site)
    if site <= first(env.bra.sites) return end

    ls = _left_nbrs_site(site, env.bra.sites)
    LL = update_env_left(
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
    RR = update_env_right(
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


function clear_env_containing_site!(env::Environment, site::Site)
    delete!(env.env, (_left_nbrs_site(site, env.ket.sites), :right))
    delete!(env.env, (_right_nbrs_site(site, env.ket.sites), :left))
end


#        -- A --
#      |    |
# L = LE -- M --  
#      |    |
#        -- B --
#

function update_env_left(LE::S, A₀::S, M::T, B₀::S) where {S <: AbstractArray{Float64, 3}, T <: AbstractDict}
    sites = sort(collect(keys(M)))
    A =_update_tensor_forward(A₀, M, sites)
    B = _update_tensor_backwards(B₀, M, sites)
    update_env_left(LE, A, M[0], B)
end


function update_env_left(LE::S, M::T) where {S <: AbstractArray{Float64, 3}, T <: AbstractDict}
    MM = M[0]  # can be more general, but it works for now
    @tensor L[nt, nc, nb] :=  LE[nt, oc, nb] * MM[oc, nc]
    L
end


function update_env_left(LE::S, A::S, M::T, B::S) where {S <: AbstractArray{Float64, 3}, T <: AbstractArray{Float64, 4}}
    # for real there is no conjugate, otherwise conj(A)
    @tensor L[nb, nc, nt] := LE[ob, oc, ot] * A[ot, α, nt] * 
                             M[oc, α, nc, β] * B[ob, β, nb] order = (ot, α, oc, β, ob)  
    L
end


function update_env_left(LE::S, A::S, M::T, B::S, ::Val{:c}) where {S <: AbstractArray{Float64, 3}, T <: AbstractArray{Float64, 4}}
    # for real there is no conjugate, otherwise conj(A)
    @tensor L[nb, nc, nt] := LE[ob, oc, ot] * A[ot, α, nt] * 
                             M[oc, β, nc, α] * B[ob, β, nb] order = (ot, α, oc, β, ob)  
    L
end


function update_env_left(LE::S, A::S, M::T, B::S) where {S <: AbstractArray{Float64, 3}, T <: SparseSiteTensor}
    L = zeros(size(B, 3), maximum(M.projs[3]), size(A, 3))

    Threads.@threads for σ ∈ 1:length(M.loc_exp)
        lexp = M.loc_exp[σ]
    #for (σ, lexp) ∈ enumerate(M.loc_exp)
        AA = @view A[:, M.projs[2][σ], :]
        LL = @view LE[:, M.projs[1][σ], :]
        BB = @view B[:, M.projs[4][σ], :]
        L[:, M.projs[3][σ], :] += lexp .* (BB' * LL * AA)
    end
    L
end


function update_env_left(LE::S, A::S, M::T, B::S, ::Val{:c}) where {S <: AbstractArray{Float64, 3}, T <: SparseSiteTensor}
    # for real there is no conjugate, otherwise conj(A)
    ## TO BE WRITTEN
end


function update_env_left(LE::S, A::S, M::T, B::S) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}
    # for real there is no conjugate, otherwise conj(A)
    ## TO BE WRITTEN
end


function update_env_left(LE::S, A::S, M::T, B::S, ::Val{:c}) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}
    # for real there is no conjugate, otherwise conj(A)
    ## TO BE WRITTEN
end


function _update_tensor_forward(A::S, M::T, sites) where {S <: AbstractArray{Float64, 3}, T <: AbstractDict}
    B = copy(A)
    for i ∈ sites
        if i == 0 break end
        C = M[i]
        @tensor B[l, x, r] := B[l, y, r] * C[y, x]
    end
    B
end


function _update_tensor_backwards(A::S, M::T, sites) where {S <: AbstractArray{Float64, 3}, T <: AbstractDict}
    B = copy(A)
    for i ∈ reverse(sites)
        if i == 0 break end
        C = M[i]
        @tensor B[l, x, r] := B[l, y, r] * C[x, y]
    end
    B
end




#      -- A --
#         |    |
# R =  -- M -- RE 
#         |    |
#      -- B --
#
function update_env_right(RE::S, A::S, M::T, B::S) where {T <: AbstractArray{Float64, 4}, S <: AbstractArray{Float64, 3}}
    # for real there is no conjugate, otherwise conj(A)
    @tensor R[nt, nc, nb] := RE[ot, oc, ob] * A[nt, α, ot] *
                             M[nc, α, oc, β] * B[nb, β, ob] order = (ot, α, oc, β, ob)
    R
end


function update_env_right(RE::S, A::S, M::T, B::S, ::Val{:c}) where {T <: AbstractArray{Float64, 4}, S <: AbstractArray{Float64, 3}}
    # for real there is no conjugate, otherwise conj(A)
    @tensor R[nt, nc, nb] := RE[ot, oc, ob] * A[nt, α, ot] * 
                             M[nc, β, oc, α] * B[nb, β, ob] order = (ot, α, oc, β, ob)
    R
end


function update_env_right(RE::S, A::S, M::T, B::S) where {T <: SparseSiteTensor, S <: AbstractArray{Float64, 3}}
    R = zeros(size(A, 1), maximum(M.projs[1]), size(B, 1))

    Threads.@threads for σ ∈ 1:length(M.loc_exp)
        lexp = M.loc_exp[σ]
    #for (σ, lexp) ∈ enumerate(M.loc_exp)
        AA = @view A[:, M.projs[2][σ], :]
        RR = @view RE[:, M.projs[3][σ], :]
        BB = @view B[:, M.projs[4][σ], :]
        R[:, M.projs[1][σ], :] += lexp .* (AA * RR * BB')
    end
    R
end


function update_env_right(RE::S, A::S, M::T, B::S, ::Val{:c}) where {T <: SparseSiteTensor, S <: AbstractArray{Float64, 3}}
    # TO BE WRITTEN
end


function update_env_right(RE::S, A::S, M::T, B::S) where {T <: SparseVirtualTensor, S <: AbstractArray{Float64,3}}
    # TO BE WRITTEN
end


function update_env_right(RE::S, A::S, M::T, B::S, ::Val{:c}) where {T <: SparseVirtualTensor, S <: AbstractArray{Float64, 3}}
    # TO BE WRITTEN
end


function update_env_right(RE::S, A₀::S, M::T, B₀::S) where {T <: AbstractDict, S <: AbstractArray{Float64, 3}}
    sites = sort(collect(keys(M)))
    A = _update_tensor_forward(A₀, M, sites)
    B = _update_tensor_backwards(B₀, M, sites)
    update_env_right(RE, A, M[0], B)
end


#           --
#              |
# R =  -- M -- RE 
#              |
#           --
#
function update_env_right(RE::S, M::T) where {S <: AbstractArray{Float64, 3}, T <: AbstractDict}
    MM = M[0]
    @tensor R[nt, nc, nb] := MM[nc, oc] * RE[nt, oc, nb]
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
function project_ket_on_bra(LE::S, B::S, M::T, RE::S) where {T <: AbstractArray{Float64, 4}, S <: AbstractArray{Float64, 3}}
    @tensor A[x, y, z] := LE[k, l, x] * B[k, m, o] * 
                          M[l, y, n, m] * RE[z, n, o] order = (k, l, m, n, o)
    A
end


function project_ket_on_bra(LE::S, B::S, M::T, RE::S) where {T <: AbstractArray{Float64, 2}, S <: AbstractArray{Float64, 3}}
    @tensor A[x, y, z] := B[x, a, z] * M[y, a]
    A
end


function project_ket_on_bra(LE::S, B::S, M::T, RE::S) where {S <: AbstractArray{Float64, 3}, T <: SparseSiteTensor}
    A = zeros(size(LE, 3), maximum(M.projs[2]), size(RE, 1))

    Threads.@threads for σ ∈ 1:length(M.loc_exp)
        lexp = M.loc_exp[σ]
    #for (σ, lexp) ∈ enumerate(M.loc_exp)
        le = @view LE[:, M.projs[1][σ], :]
        b = @view B[:, M.projs[4][σ], :]
        re = @view RE[:, M.projs[3][σ], :]
        A[:, M.projs[2][σ], :] += lexp .* (le' * b * re')
    end
    A
end


function project_ket_on_bra(LE::S, B::S, M::T, RE::S) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}
    # TO BE WRITTEN
end


function project_ket_on_bra(LE::S, B::S, M::T, RE::S, ::Val{:c}) where {T <: AbstractArray{Float64, 4}, S <: AbstractArray{Float64, 3}}
    @tensor A[x, y, z] := LE[k, l, x] * B[k, m, o] * 
                          M[l, m, n, y] * RE[z, n, o] order = (k, l, m, n, o)
    A
end


function project_ket_on_bra(LE::S, B::S, M::T, RE::S, ::Val{:c}) where {S <: AbstractArray{Float64, 3}, T <: SparseSiteTensor}
    ## TO BE WRITTEN
end


function project_ket_on_bra(LE::S, B::S, M::T, RE::S, ::Val{:c}) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}
    ## TO BE ADDED
end


function project_ket_on_bra(LE::S, B₀::S, M::T, RE::S) where {S <: AbstractArray{Float64, 3}, T <: AbstractDict}
    sites = sort(collect(keys(M)))
    C = sort(collect(M), by = x->x[1])
    TT = B₀
    for (t, v) ∈ reverse(C) TT = project_ket_on_bra(LE, TT, v, RE) end
    TT
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
    for i ∈ reverse(ψ.sites)
        B = ψ.tensors[i]
        @matmul M̃[x, (σ, y)] := sum(α) B[x, σ, α] * R[α, y]
        R, Q = rq_fact(M̃, Dcut, args...)
        R = R ./ maximum(abs.(R))
        @cast B[x, σ, y] := Q[x, (σ, y)] (σ ∈ 1:size(B, 2))
        ψ.tensors[i] = B
    end
end


