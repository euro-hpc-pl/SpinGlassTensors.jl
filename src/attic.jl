"""
$(TYPEDSIGNATURES)
"""
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
            env.bra[env.bra.sites[1]] = US / norm(US)
            update_env_right!(env, env.bra.sites[2])
            update_env_right!(env, env.bra.sites[1])
        end
    end
end

#TODO: to be changed
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

# TODO: doesn't work
function compress_twosite!(
    bra::QMps, mpo::QMpo, ket::QMps, Dcut::Int, tol::Real=1E-8, max_sweeps::Int=4, args...
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
