export
    variational_compress!,
    _left_nbrs_site,
    _right_nbrs_site,
    canonise!,
    canonise_truncate!,
    truncate!,
    variational_sweep!,
    Environment,
    projectors_to_sparse,
    projectors_to_sparse_transposed

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

"""
Largest x in sites: x < site
"""
function _left_nbrs_site(site::Site, sites)
    ls = filter(i -> i < site, sites)
    if isempty(ls) return -Inf end
    maximum(ls)
end

"""
Smallest x in sites: x > site
"""
function _right_nbrs_site(site::Site, sites)
    ms = filter(i -> i > site, sites)
    if isempty(ms) return Inf end
    minimum(ms)
end

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

function clear_env_containing_site!(env::Environment, site::Site)
    delete!(env.env, (_left_nbrs_site(site, env.ket.sites), :right))
    delete!(env.env, (_right_nbrs_site(site, env.ket.sites), :left))
end

"""
        -- A --
      |    |
 L = LE -- M --
      |    |
        -- B --
"""
function update_env_left(
    LE::S, A₀::S, M::T, B₀::S, trans::Symbol=:n
) where {S <: ArrayOrCuArray{3}, T <: AbstractDict}
    sites = sort(collect(keys(M)))
    A = _update_tensor_forward(A₀, M, sites, Val(trans))
    B = _update_tensor_backwards(B₀, M, sites, Val(trans))
    update_env_left(LE, A, M[0], B, Val(trans))
end

function update_env_left(
    LE::S, M::T, trans::Symbol=:n
) where {S <: ArrayOrCuArray{3}, T <: AbstractDict}
    attach_central_left(LE, M[0])
end

projector_to_dense(::Type{T}, pr::Array{Int, 1}) where T = diagm(ones(T, maximum(pr)))[:, pr]
projector_to_dense(pr::Array{Int, 1}) = projector_to_dense(Float64, pr)

function projectors_to_sparse(p_lb::Array{Int, 1}, p_l::Array{Int, 1}, p_lt::Array{Int, 1}, env)
    if env <: CUDA.CuArray
        ps = projectors_to_sparse(p_lb, p_l, p_lt, Val(:cs))
    else
        ps = projectors_to_sparse(p_lb, p_l, p_lt, Val(:s))
    end
    ps
end

function projectors_to_sparse_transposed(p_lb::Array{Int, 1}, p_l::Array{Int, 1}, p_lt::Array{Int, 1}, env)
    if env <: CUDA.CuArray
        ps = projectors_to_sparse_transposed(p_lb, p_l, p_lt, Val(:cs))
    else
        ps = projectors_to_sparse_transposed(p_lb, p_l, p_lt, Val(:s))
    end
    ps
end

function projectors_to_sparse(p_lb::Array{Int, 1}, p_l::Array{Int, 1}, p_lt::Array{Int, 1}, ::Val{:s})
    # asumption length(p_lb) == length(p_l) == length(p_lt)
    columns = length(p_lb)
    temp = Vector{Int64}()
    ps_vect = Vector{Int64}()

    # @cast temp[x,y,w] = p_lb[x, w] * p_l[y,w]
    # reshape(temp, (x*y, w))
    rows_p_lb = maximum(p_lb)
    for i ∈ collect(1:columns)
        push!(temp, rows_p_lb*(p_l[i] -1) + p_lb[i])
    end

    # @cast ps_vect[x,y,z, w] = p_lb[x, w] * p_l[y,w] * p_lt[z,w] = temp[x, y, w] * p_lt[z, w]
     # reshape(ps_vect, (x*y*z, w))
    temp_rows = maximum(p_lb) * maximum(p_l)
    for i ∈ collect(1:columns)
        push!(ps_vect, temp_rows*(p_lt[i] -1) + temp[i])
    end

    rowInd = ps_vect
    colInd = collect(1:columns)
    Values = ones(Float64, columns)
    ps = sparse(rowInd, colInd, Values, temp_rows*maximum(p_lt), columns)
    ps
end

function projectors_to_sparse(p_lb::Array{Int, 1}, p_l::Array{Int, 1}, p_lt::Array{Int, 1}, ::Val{:cs})
    # asumption length(p_lb) == length(p_l) == length(p_lt)
    p_l = CUDA.CuArray(p_l)
    p_lb = CUDA.CuArray(p_lb)
    p_lt = CUDA.CuArray(p_lt)

    rowPtr = maximum(p_l) * maximum(p_lb) * (p_lt .- 1) .+ maximum(p_lb) * (p_l .- 1) .+ p_lb

    columns = length(p_lb)
    rows = maximum(p_l) * maximum(p_lb) * maximum(p_lt)
    colPtr = CUDA.CuArray(collect(1:columns+1))
    nzVal = CUDA.ones(Float64, columns)

    CUSPARSE.CuSparseMatrixCSC(colPtr, rowPtr, nzVal, (rows, columns))
end

function projectors_to_sparse_transposed(p_lb::Array{Int, 1}, p_l::Array{Int, 1}, p_lt::Array{Int, 1}, ::Val{:cs})
    p_l = CUDA.CuArray(p_l)
    p_lb = CUDA.CuArray(p_lb)
    p_lt = CUDA.CuArray(p_lt)

    rowPtr = maximum(p_l) * maximum(p_lb) * (p_lt .- 1) .+ maximum(p_lb) * (p_l .- 1) .+ p_lb

    columns = length(p_lb)
    rows = maximum(p_l) * maximum(p_lb) * maximum(p_lt)
    colPtr = CUDA.CuArray(collect(1:columns+1))
    nzVal = CUDA.ones(Float64, columns)

    CUSPARSE.CuSparseMatrixCSR(colPtr, rowPtr, nzVal, (columns, rows))
end


function _update_tensor_forward(
    A::S, M::T, sites, ::Val{:n}
) where {S <: ArrayOrCuArray{3}, T <: AbstractDict}
    B = copy(A)
    for i ∈ sites
        if i == 0 break end
        C = M[i]
        B = attach_central_left(B, C)
    end
    B
end

function _update_tensor_forward(
    A::S, M::T, sites, ::Val{:c}
) where {S <: ArrayOrCuArray{3}, T <: AbstractDict}
    B = copy(A)
    for i ∈ reverse(sites)
        if i == 0 break end
        C = M[i]
        B = attach_central_right(B, C)
    end
    B
end

function _update_tensor_backwards(
    A::S, M::T, sites, ::Val{:n}
) where {S <: ArrayOrCuArray{3}, T <: AbstractDict}
    B = copy(A)
    for i ∈ reverse(sites)
        if i == 0 break end
        C = M[i]
        B = attach_central_right(B, C)
    end
    B
end

function _update_tensor_backwards(
    A::S, M::T, sites, ::Val{:c}
) where {S <: ArrayOrCuArray{3}, T <: AbstractDict}
    B = copy(A)
    for i ∈ sites
        if i == 0 break end
        C = M[i]
        B = attach_central_left(B, C)
    end
    B
end

function update_env_right(
    RE::S, A₀::S1, M::T, B₀::S, trans::Symbol
) where {T <: AbstractDict, S <: ArrayOrCuArray{3}, S1 <: ArrayOrCuArray{3}}
    sites = sort(collect(keys(M)))
    A = _update_tensor_forward(A₀, M, sites, Val(trans))
    B = _update_tensor_backwards(B₀, M, sites, Val(trans))
    update_env_right(RE, A, M[0], B, Val(trans))
end

"""
           --
              |
 R =  -- M -- RE
              |
           --
"""
function update_env_right(
    RE::S, M::T, trans::Symbol
) where {S <: ArrayOrCuArray{3}, T <: AbstractDict}
    attach_central_right(RE, M[0])
end

function project_ket_on_bra(env::Environment, site::Site, trans::Symbol)
    project_ket_on_bra(
        env.env[(site, :left)],
        env.ket[site],
        env.mpo[site],
        env.env[(site, :right)],
        Val(trans)
    )
end

function project_ket_on_bra(
    LE::S, B₀::S, M::T, RE::S, ::Val{:n}
) where {S <: ArrayOrCuArray{3}, T <: AbstractDict}
    C = sort(collect(M), by = x -> x[1])
    TT = B₀
    for (_, v) ∈ reverse(C)
        dimv = length(size(v))
        if dimv == 2
            TT = attach_central_right(TT, v)
        else
            TT = project_ket_on_bra(LE, TT, v, RE, Val(:n))
        end
    end
    TT
end

function project_ket_on_bra(
    LE::S, B₀::S, M::T, RE::S, ::Val{:c}
) where {S <: ArrayOrCuArray{3}, T <: AbstractDict}
    C = sort(collect(M), by = x -> x[1])
    TT = B₀
    for (_, v) ∈ C
        dimv = length(size(v))
        if dimv == 2
            TT = attach_central_left(TT, v)
        else
            TT = project_ket_on_bra(LE, TT, v, RE, Val(:c))
        end
    end
    TT
end

function measure_env(env::Environment, site::Site, trans::Symbol)
    L = update_env_left(
        env.env[(site, :left)], env.bra[site], env.mpo[site], env.ket[site], trans
    )
    R = env.env[(site, :right)]
    @tensor L[t, c, b] * R[b, c, t]
end

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

canonise!(ψ::QMps, s::Symbol) = canonise!(ψ, Val(s))
canonise!(ψ::QMps, ::Val{:right}) = _left_sweep!(ψ, typemax(Int))
canonise!(ψ::QMps, ::Val{:left}) = _right_sweep!(ψ, typemax(Int))

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
