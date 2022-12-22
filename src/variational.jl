export
    variational_compress!,
    variational_sweep!,
    zipper

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

function variational_sweep!(   # we may be able to remove it
    bra::QMps{T}, mpo::QMpo{T}, ket::QMps{T}, ::Val{:left}, args...
) where T <: Real
    env = Environment(bra, mpo, ket)
    _right_sweep_var!(env, args...)
end

function variational_sweep!(   # we may be able to remove it
    bra::QMps{T}, mpo::QMpo{T}, ket::QMps{T}, ::Val{:right}, args...
) where T <: Real
    env = Environment(bra, mpo, ket)
    _left_sweep_var!(env, args...)
end

function zipper(ψ::QMpo{R}, ϕ::QMps{R}, Dcut::Int=typemax(Int), tol::Real=eps(), args...) where R <: Real
    # input ϕ should be canonized :left
    # results should be canonized :right
    D = TensorMap{R}()
    T = ones(R, 1, 1, 1)
    mpo_li = last(ψ.sites)
    for i ∈ reverse(ϕ.sites)
        while mpo_li > i
            T = contract_matrix_tensor3(ψ[mpo_li], T)
            mpo_li = left_nbrs_site(mpo_li, ψ.sites)
        end
        @assert mpo_li == i "Mismatch between QMpo and QMps sites."
        mpo_li = left_nbrs_site(mpo_li, ψ.sites)

        M, B = ψ[i], ϕ[i]
        for v ∈ reverse(M.bot) B = contract_matrix_tensor3(v, B) end

        @tensor TT[l, ml, mt, tt] := M.ctr[ml, mt, mr, mb] * B[l, mb, r] *
                                     T[tt, mr, r] order = (r, mb, mr)
        s1, s2, s3, s4 = size(TT)
        @cast TT[(t1, t2), t3, t4] := TT[t1, t2, t3, t4]
        for v ∈ reverse(M.top) TT = contract_matrix_tensor3(v, TT) end
        @cast TT[t12, (t3, t4)] := TT[t12, t3, t4]

        U, Σ, V = svd(TT, Dcut, tol, args...)
        V = Array(V') # ?
        T = U * diagm(Σ)
        if i == ϕ.sites[1] V = T * V end
        @cast V[x, y, z] := V[x, (y, z)] (y ∈ 1:s3)
        @cast T[x, y, z] := T[(x, y), z] (y ∈ 1:s2)
        T = permutedims(T, (3, 2, 1))
        push!(D, i => V)
    end
    QMps(D)
end
