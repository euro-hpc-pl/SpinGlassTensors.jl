# gauges.jl: This file provides basic functions to optimize gauges for the PEPS network. CUDA is supported.

export optimize_gauges_for_overlaps!!, overlap_density_matrix

function update_rq!(ψ::QMps{T}, AT::AbstractArray{T,3}, i::Site) where {T<:Real}
    # @cast ATR[x, (y, σ)] := AT[x, y, σ]
    toGPU = ψ.onGPU ? true : false

    ATR = reshape(AT, size(AT, 1), size(AT, 2) * size(AT, 3))
    RT, QT = rq_fact(ATR; toGPU=toGPU)
    RT ./= maximum(abs.(RT))
    # @cast AT[x, y, σ] := QT[x, (y, σ)] (σ ∈ 1:size(AT, 3))
    AT = reshape(QT, size(QT, 1), size(QT, 2) ÷ size(AT, 3), size(AT, 3))

    ψ[i] = AT
    RT
end

function update_qr!(ψ::QMps{T}, AT::AbstractArray{T,3}, i::Site) where {T<:Real}
    # @cast ATR[(x, σ), y] := AT[x, y, σ]
    toGPU = ψ.onGPU ? true : false

    ATR = reshape(AT, size(AT, 1) * size(AT, 3), size(AT, 2))
    QT, RT = qr_fact(ATR; toGPU=toGPU)
    RT ./= maximum(abs.(RT))
    # @cast AT[x, y, σ] := QT[(x, σ), y] (σ ∈ 1:size(AT, 3))
    AT = reshape(QT, size(QT, 1) ÷ size(AT, 3), size(QT, 2), size(AT, 3))

    ψ[i] = AT
    RT
end

# function _gauges_right_sweep!!!(
#     ψ_top::QMps{R},
#     ψ_bot::QMps{R},
#     gauges::Dict;
#     tol = 1E-12,
# ) where {R<:Real}
#     RT = ψ_top.onGPU && ψ_bot.onGPU ? CUDA.ones(R, 1, 1) : ones(R, 1, 1)
#     # println(ψ_top.onGPU)
#     RB = copy(RT)
#     for i ∈ ψ_top.sites
#         T, B = ψ_top[i], ψ_bot[i]

#         @tensor T[a, b, c] := RT[a, s] * T[s, b, c]
#         @tensor B[a, b, c] := RB[a, s] * B[s, b, c]
#         @tensor ρ_t[r, s] := T[i, j, r] * conj(T)[i, j, s]
#         @tensor ρ_b[r, s] := B[i, j, r] * conj(B)[i, j, s]

#         dρ_b, dρ_t = diag.((ρ_b, ρ_t))
#         K = (dρ_b .< tol) .|| (dρ_t .< tol)
#         dρ_b[K] .= 1
#         dρ_t[K] .= 1

#         X = (dρ_b ./ dρ_t) .^ (1 / 4) # optimize
#         X_inv = 1 ./ X
#         gauges[i] .*= X  # update

#         RT = update_qr!(ψ_top, T .* reshape(X, 1, 1, :), i)
#         RB = update_qr!(ψ_bot, B .* reshape(X_inv, 1, 1, :), i)

#     end
# end

# function _gauges_left_sweep!!!(
#     ψ_top::QMps{R},
#     ψ_bot::QMps{R},
#     gauges::Dict;
#     tol = 1E-12,
# ) where {R<:Real}
#     RT = ψ_top.onGPU && ψ_bot.onGPU ? CUDA.ones(R, 1, 1) : ones(R, 1, 1)
#     # println(ψ_top.onGPU)
#     RB = copy(RT)
#     for i ∈ reverse(ψ_top.sites)
#         T, B = ψ_top[i], ψ_bot[i]

#         @tensor T[a, c, b] := T[a, s, b] * RT[s, c]
#         @tensor B[a, c, b] := B[a, s, b] * RB[s, c]
#         @tensor ρ_t[r, s] := T[i, j, r] * conj(T)[i, j, s]
#         @tensor ρ_b[r, s] := B[i, j, r] * conj(B)[i, j, s]

#         dρ_b, dρ_t = diag.((ρ_b, ρ_t))
#         K = (dρ_b .< tol) .|| (dρ_t .< tol)
#         dρ_b[K] .= 1
#         dρ_t[K] .= 1

#         X = (dρ_b ./ dρ_t) .^ (1 / 4) # optimize
#         X_inv = 1 ./ X
#         gauges[i] .*= X # update

#         RT = update_rq!(ψ_top, T .* reshape(X, 1, 1, :), i)
#         RB = update_rq!(ψ_bot, B .* reshape(X_inv, 1, 1, :), i)
#     end
# end

function _gauges_right_sweep!!!(
    ψ_top::QMps{R},
    ψ_bot::QMps{R},
    gauges::Dict;
    tol = 1E-12
) where {R<:Real}
    RT = ψ_top.onGPU && ψ_bot.onGPU ? CUDA.ones(R, 1, 1) : ones(R, 1, 1)
    RB = copy(RT)

    for i ∈ ψ_top.sites
        T, B = ψ_top[i], ψ_bot[i]
        T = batched_mul(RT, T) 
        B = batched_mul(RB, B)

        T_reshaped = reshape(T, :, size(T, 3)) 
        conjT_reshaped = reshape(conj(T), :, size(T, 3)) 

        B_reshaped = reshape(B, :, size(B, 3)) 
        conjB_reshaped = reshape(conj(B), :, size(B, 3)) 

        dρ_t = sum(T_reshaped .* conjT_reshaped, dims=1)
        dρ_b = sum(B_reshaped .* conjB_reshaped, dims=1)

        dρ_t = dropdims(dρ_t, dims=1) 
        dρ_b = dropdims(dρ_b, dims=1) 

        K = (dρ_b .< tol) .|| (dρ_t .< tol)
        dρ_b[K] .= 1
        dρ_t[K] .= 1

        X = (dρ_b ./ dρ_t) .^ (1 / 4)
        X_inv = 1 ./ X

        gauges[i] .*= X
        RT = update_qr!(ψ_top, T .* reshape(X, 1, 1, :), i)
        RB = update_qr!(ψ_bot, B .* reshape(X_inv, 1, 1, :), i)
    end
end

function _gauges_left_sweep!!!(
    ψ_top::QMps{R},
    ψ_bot::QMps{R},
    gauges::Dict;
    tol = 1E-12
) where {R<:Real}
    RT = ψ_top.onGPU && ψ_bot.onGPU ? CUDA.ones(R, 1, 1) : ones(R, 1, 1)
    RB = copy(RT)

    for i ∈ reverse(ψ_top.sites)
        T, B = ψ_top[i], ψ_bot[i]

        T_reshaped = reshape(T, size(T, 1), :, size(T, 3))
        T = batched_mul(T_reshaped, RT)
        B_reshaped = reshape(B, size(B, 1), :, size(B, 3)) 
        B = batched_mul(B_reshaped, RB)

        T_reshaped = reshape(T, :, size(T, 3)) 
        conjT_reshaped = reshape(conj(T), :, size(T, 3)) 

        dρ_t = sum(T_reshaped .* conjT_reshaped, dims=1) 
        dρ_t = dropdims(dρ_t, dims=1) 

        B_reshaped = reshape(B, :, size(B, 3)) 
        conjB_reshaped = reshape(conj(B), :, size(B, 3)) 

        dρ_b = sum(B_reshaped .* conjB_reshaped, dims=1) 
        dρ_b = dropdims(dρ_b, dims=1) 

        K = (dρ_b .< tol) .|| (dρ_t .< tol) 
        dρ_b[K] .= 1
        dρ_t[K] .= 1

        X = (dρ_b ./ dρ_t) .^ (1 / 4) 
        X_inv = 1 ./ X
        gauges[i] .*= X 

        RT = update_rq!(ψ_top, T .* reshape(X, 1, 1, :), i)
        RB = update_rq!(ψ_bot, B .* reshape(X_inv, 1, 1, :), i)
    end
end

function optimize_gauges_for_overlaps!!(
    ψ_top::QMps{T},
    ψ_bot::QMps{T},
    tol = 1E-8,
    max_sweeps::Int = 4,
) where {T<:Real}
    onGPU = ψ_top.onGPU && ψ_bot.onGPU
    canonise!(ψ_top, :left)
    canonise!(ψ_bot, :left)
    
    overlap_old = dot(ψ_top, ψ_bot)
    gauges = Dict(i => (onGPU ? CUDA.ones : ones)(T, size(ψ_top[i], 3)) for i ∈ ψ_top.sites)
    for i ∈ 1:max_sweeps
        _gauges_right_sweep!!!(ψ_top, ψ_bot, gauges)
        _gauges_left_sweep!!!(ψ_top, ψ_bot, gauges)

        overlap_new = dot(ψ_top, ψ_bot)
        Δ = overlap_new / overlap_old
        overlap_old = overlap_new
        if abs(Δ - one(T)) < tol
            break
        end
    end
    gauges
end

function overlap_density_matrix(ϕ::QMps{T}, ψ::QMps{T}, k::Site) where {T<:Real}
    @assert ψ.sites == ϕ.sites
    C = _overlap_forward(ϕ, ψ, k)
    D = _overlap_backwards(ϕ, ψ, k)
    A, B = ψ[k], ϕ[k]
    @tensor E[x, y] := C[b, a] * conj(B)[b, β, x] * A[a, α, y] * D[β, α]
end

function _overlap_forward(ϕ::QMps{T}, ψ::QMps{T}, k::Site) where {T<:Real}
    C = ϕ.onGPU && ψ.onGPU ? CUDA.ones(T, 1, 1) : ones(T, 1, 1)
    i = ψ.sites[1]
    while i < k
        A, B = ψ[i], ϕ[i]
        @tensor order = (α, β, σ) C[x, y] := conj(B)[β, x, σ] * C[β, α] * A[α, y, σ]
        i += 1
    end
    C
end

function _overlap_backwards(ϕ::QMps{T}, ψ::QMps{T}, k::Site) where {T<:Real}
    D = ϕ.onGPU && ψ.onGPU ? CUDA.ones(T, 1, 1) : ones(T, 1, 1)
    i = ψ.sites[end]
    while i > k
        A, B = ψ[i], ϕ[i]
        @tensor order = (α, β, σ) D[x, y] := conj(B)[x, β, σ] * D[β, α] * A[y, α, σ]
        i -= 1
    end
    D
end
