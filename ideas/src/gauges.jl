export
    GaugeOptimizer

struct GaugeOptimizer{T}
    tol::T
    max_sweeps::Int

    GaugeOptimizer(; tol=1E-8, max_sweeps=4) = new{eltype(tol)}(tol, max_sweeps)
end

function (optimizer::GaugeOptimize{T})(ψ_top::QMps{T}, ψ_bot::QMps{T}) where T
    canonise!(ψ_top, :right)
    canonise!(ψ_bot, :right)

    overlap_old = dot(ψ_top, ψ_bot)
    gauges = Dict(i => ones(T, size(ψ_top[i], 2)) for i ∈ ψ_top.sites)
    for _ ∈ 1:optimizer.max_sweeps
        _gauges_right_sweep!!!(ψ_top, ψ_bot, gauges)
        _gauges_left_sweep!!!(ψ_top, ψ_bot, gauges)

        overlap_new = dot(ψ_top, ψ_bot)
        Δ = overlap_new / overlap_old
        overlap_old = overlap_new
        abs(Δ - one(T)) < optimizer.tol && break
    end
    gauges
end

function _gauges_right_sweep!!!(ψ_top::QMps{R}, ψ_bot::QMps{R}, all_gauges::Dict; tol=1e-12) where R
    RT = ones(R, 1, 1)
    RB = ones(R, 1, 1)
    for i ∈ ψ_top.sites
        T, B = ψ_top[i], ψ_bot[i]

        @tensor T[a, b, c] := RT[a, s] * T[s, b, c]
        @tensor B[a, b, c] := RB[a, s] * B[s, b, c]
        @tensor ρ_t[r, s] := T[i, r, j] * conj(T)[i, s, j]
        @tensor ρ_b[r, s] := B[i, r, j] * conj(B)[i, s, j]

        dρ_b = diag(ρ_b)
        dρ_t = diag(ρ_t)
        inds = (dρ_b .< ep) .|| (dρ_t .< tol)
        dρ_b[inds] .= one(R)
        dρ_t[inds] .= one(R)

        gauge = (dρ_b ./ dρ_t) .^ (1 / 4) # optimize
        gauge_inv = one(R) ./ gauge
        all_gauges[i] .*= gauge # update

        AT = T .* reshape(gauge, (1, :, 1))
        AB = B .* reshape(gauge_inv, (1, :, 1))

        @cast ATR[(x, σ), y] := AT[x, σ, y]
        QT, RT = qr_fact(ATR)
        RT ./= maximum(abs.(RT))
        @cast AT[x, σ, y] := QT[(x, σ), y] (σ ∈ 1:size(AT, 2))
        ψ_top[i] = AT

        @cast ABR[(x, σ), y] := AB[x, σ, y]
        QB, RB = qr_fact(ABR)
        RB ./= maximum(abs.(RB))
        @cast AB[x, σ, y] := QB[(x, σ), y] (σ ∈ 1:size(AB, 2))
        ψ_bot[i] = AB
    end
end

function _gauges_left_sweep!!!(ψ_top::QMps{R}, ψ_bot::QMps{R}, all_gauges::Dict; tol=1e-12) where R
    RT = ones(R, 1, 1)
    RB = ones(R, 1, 1)

    for i ∈ reverse(ψ_top.sites)
        T, B = ψ_top[i], ψ_bot[i]

        @tensor T[a, b, c] := T[a, b, s] * RT[s, c]
        @tensor B[a, b, c] := B[a, b, s] * RB[s, c]
        @tensor ρ_t[r, s] := T[i, r, j] * conj(T)[i, s, j]
        @tensor ρ_b[r, s] := B[i, r, j] * conj(B)[i, s, j]

        dρ_b = diag(ρ_b)
        dρ_t = diag(ρ_t)
        inds = (dρ_b .< ep) .|| (dρ_t .< tol)
        dρ_b[inds] .= one(R)
        dρ_t[inds] .= one(R)

        gauge = (dρ_b ./ dρ_t) .^ (1 / 4) # optimize

        gauge_inv = one(R) ./ gauge
        all_gauges[i] .*= gauge # update

        AT = T .* reshape(gauge, (1, :, 1))
        AB = B .* reshape(gauge_inv, (1, :, 1))

        @cast ATR[x, (σ, y)] := AT[x, σ, y]
        RT, QT = rq_fact(ATR)
        RT ./= maximum(abs.(RT))
        @cast AT[x, σ, y] := QT[x, (σ, y)] (σ ∈ 1:size(AT, 2))
        ψ_top[i] = AT

        @cast ABR[x, (σ, y)] := AB[x, σ, y]
        RB, QB = rq_fact(ABR)
        RB ./= maximum(abs.(RB))
        @cast AB[x, σ, y] := QB[x, (σ, y)] (σ ∈ 1:size(AB, 2))
        ψ_bot[i] = AB
    end
end
