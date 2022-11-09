export
    optimize_gauges_for_overlaps!!

function _gauges_right_sweep!!!(ψ_top::QMps, ψ_bot::QMps, all_gauges::Dict)
    RT, RB = ones(1, 1), ones(1, 1)
    for i ∈ ψ_top.sites
        T, B = ψ_top[i], ψ_bot[i]

        @tensor T[a, b, c] := RT[a, s] * T[s, b, c]
        @tensor B[a, b, c] := RB[a, s] * B[s, b, c]
        @tensor ρ_t[r, s] := T[i, r, j] * conj(T)[i, s, j]
        @tensor ρ_b[r, s] := B[i, r, j] * conj(B)[i, s, j]

        dρ_b = diag(ρ_b)
        dρ_t = diag(ρ_t)
        ep = 1e-12
        inds = (dρ_b .< ep) .|| (dρ_t .< ep)
        dρ_b[inds] .= 1.0
        dρ_t[inds] .= 1.0

        gauge = (dρ_b ./ dρ_t) .^ (1 / 4) # optimize
        gauge_inv = 1.0 ./ gauge
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

function _gauges_left_sweep!!!(ψ_top::QMps, ψ_bot::QMps, all_gauges::Dict)
    RT, RB = ones(1, 1), ones(1, 1)

    for i ∈ reverse(ψ_top.sites)
        T, B = ψ_top[i], ψ_bot[i]

        @tensor T[a, b, c] := T[a, b, s] * RT[s, c]
        @tensor B[a, b, c] := B[a, b, s] * RB[s, c]
        @tensor ρ_t[r, s] := T[i, r, j] * conj(T)[i, s, j]
        @tensor ρ_b[r, s] := B[i, r, j] * conj(B)[i, s, j]

        dρ_b = diag(ρ_b)
        dρ_t = diag(ρ_t)
        ep = 1e-12
        inds = (dρ_b .< ep) .|| (dρ_t .< ep)
        dρ_b[inds] .= 1.0
        dρ_t[inds] .= 1.0

        gauge = (dρ_b ./ dρ_t) .^ (1 / 4) # optimize

        gauge_inv = 1.0 ./ gauge
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

function optimize_gauges_for_overlaps!!(
    ψ_top::QMps,
    ψ_bot::QMps,
    tol::Real=1E-8,
    max_sweeps::Int=4
)
    canonise!(ψ_top, :right)
    canonise!(ψ_bot, :right)

    overlap_old = dot(ψ_top, ψ_bot)
    all_gauges = Dict(i => ones(size(ψ_top[i], 2)) for i ∈ ψ_top.sites)
    for _ ∈ 1:max_sweeps
        _gauges_right_sweep!!!(ψ_top, ψ_bot, all_gauges)
        _gauges_left_sweep!!!(ψ_top, ψ_bot, all_gauges)

        overlap_new = dot(ψ_top, ψ_bot)
        Δ = overlap_new / overlap_old
        overlap_old = overlap_new
        if abs(Δ - 1.0) < tol break end
    end
    all_gauges
end
