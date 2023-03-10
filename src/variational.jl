

# variational.jl: This file provides basic functions to perform variational compression for MPS.
#                 If the MPS is moved to the GPU, its compression will be performed on the device.

export
    variational_compress!,
    variational_sweep!

function variational_compress!(bra::QMps{T}, mpo::QMpo{T}, ket::QMps{T}, tol=1E-10, max_sweeps::Int=4, kwargs...) where T <: Real
    @assert is_left_normalized(bra)
    @assert is_left_normalized(ket)
    env = Environment(bra, mpo, ket)
    overlap = Inf
    overlap_0, negative = measure_env(env, last(env.bra.sites))
    if negative
        env.bra[last(env.bra.sites)] .*= -1
    end
    println(" sweep = 0 overlap = ", overlap_0)

    for sweep ∈ 1:max_sweeps
        _left_sweep_var!(env; kwargs...)
        _right_sweep_var!(env; kwargs...)
        overlap, negative = measure_env(env, last(env.bra.sites))
        if negative
            env.bra[last(env.bra.sites)] .*= -1
        end
        Δ = abs(overlap_0 - overlap)
        @info "Convergence" Δ
        if Δ < tol
            @info "Finished in $sweep sweeps of $(max_sweeps)."
            return overlap, env
        else
            overlap_0 = overlap
        end
        println(" sweep = ", sweep , " overlap = ", overlap, " Delta = ", Δ)
    end
    println("Memory bra = ", format_bytes.(measure_memory(bra)), " mpo = ", format_bytes.(measure_memory(mpo)), " env = ", format_bytes.(measure_memory(env)))
    overlap, env
end

function _left_sweep_var!(env::Environment; kwargs...)
    toGPU = env.ket.onGPU && env.mpo.onGPU && env.bra.onGPU
    for site ∈ reverse(env.bra.sites)
        update_env_right!(env, site)
        A = project_ket_on_bra(env, site)
        @cast B[l, (r, t)] := A[l, r, t]
        _, Q = rq_fact(B; toGPU = toGPU, kwargs...)
        @cast C[l, r, t] := Q[l, (r, t)] (t ∈ 1:size(A, 3))
        env.bra[site] = C
        clear_env_containing_site!(env, site)
    end
end

function _right_sweep_var!(env::Environment; kwargs...)
    toGPU = env.ket.onGPU && env.mpo.onGPU && env.bra.onGPU
    for site ∈ env.bra.sites
        update_env_left!(env, site)
        A = project_ket_on_bra(env, site)
        B = permutedims(A, (1, 3, 2))  # [l, t, r]
        @cast B[(l, t), r] := B[l, t, r]
        Q, _ = qr_fact(B; toGPU = toGPU, kwargs...)
        @cast C[l, t, r] := Q[(l, t), r] (t ∈ 1:size(A, 3))
        C = permutedims(C, (1, 3, 2))  # [l, r, t]
        env.bra[site] = C
        clear_env_containing_site!(env, site)
    end
end

# TODO those 2 functions are to be removed eventually
function variational_sweep!(bra::QMps{T}, mpo::QMpo{T}, ket::QMps{T}, ::Val{:left}; kwargs...) where T <: Real
    _right_sweep_var!(Environment(bra, mpo, ket); kwargs...)
end

function variational_sweep!(bra::QMps{T}, mpo::QMpo{T}, ket::QMps{T}, ::Val{:right}; kwargs...) where T <: Real
    _left_sweep_var!(Environment(bra, mpo, ket); kwargs...)
end
