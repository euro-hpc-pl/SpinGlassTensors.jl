export
    variational_compress!,
    variational_sweep!

function variational_compress!(
    bra::QMps{T}, mpo::QMpo{T}, ket::QMps{T}, tol::Real=1E-10, max_sweeps::Int=4, args...
) where T <: Real
    env = Environment(bra, mpo, ket)
    overlap = Inf
    overlap_0 = measure_env(env, last(env.bra.sites))

    for sweep ∈ 1:max_sweeps
        _left_sweep_var!(env, args...)
        _right_sweep_var!(env, args...)

        overlap = measure_env(env, last(env.bra.sites))
        Δ = abs((overlap_0 - overlap) / overlap)
        @info "Convergence" Δ

        if Δ < tol
            @info "Finished in $sweep sweeps of $(max_sweeps)."
            return overlap
        else
            overlap_0 = overlap
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

function variational_sweep!(   # TODO we may be able to remove it
    bra::QMps{T}, mpo::QMpo{T}, ket::QMps{T}, ::Val{:left}, args...
) where T <: Real
    env = Environment(bra, mpo, ket)
    _right_sweep_var!(env, args...)
end

function variational_sweep!(   # TODO we may be able to remove it
    bra::QMps{T}, mpo::QMpo{T}, ket::QMps{T}, ::Val{:right}, args...
) where T <: Real
    env = Environment(bra, mpo, ket)
    _left_sweep_var!(env, args...)
end
