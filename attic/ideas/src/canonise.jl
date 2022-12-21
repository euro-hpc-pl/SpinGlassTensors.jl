export
    canonise!,
    truncate!,
    canonise_truncate!

canonise!(ψ::QMps, s::Symbol) = canonise!(ψ, Val(s))
canonise!(ψ::QMps, ::Val{:right}) = _left_sweep!(ψ, typemax(Int))
canonise!(ψ::QMps, ::Val{:left}) = _right_sweep!(ψ, typemax(Int))

function canonise_truncate!(ψ::QMps, dir::Symbol; Dcut::Int=typemax(Int), tolS::Real=eps(), args...)
    @assert dir ∈ (:left, :right)
    (dir == :right ? _left_sweep! : _right_sweep!)(ψ, Dcut, tolS, args...)
end

function truncate!(ψ::QMps, dir::Symbol; Dcut::Int=typemax(Int), tolS::Real=eps(), args...)
    @assert s ∈ (:left, :right)
    if dir == :right
        _right_sweep!(ψ, args...)
        _left_sweep!(ψ, Dcut, tolS, args...)
    else
        _left_sweep!(ψ, args...)
        _right_sweep!(ψ, Dcut, tolS, args...)
    end
end

function _right_sweep!(ψ::QMps{T}, Dcut::Int=typemax(Int), tolS::Real=eps(), args...) where T
    R = ones(T, 1, 1)
    for i ∈ ψ.sites
        A = ψ[i]
        @matmul M[(x, σ), y] := sum(α) R[x, α] * A[α, σ, y]
        Q, R = qr_fact(M, Dcut, tolS, args...)
        R ./= maximum(abs.(R))
        @cast A[x, σ, y] := Q[(x, σ), y] (σ ∈ 1:size(A, 2))
        ψ[i] = A
    end
end

function _left_sweep!(ψ::QMps{T}, Dcut::Int=typemax(Int), tolS::Real=eps(), args...) where T
    R = ones(T, 1, 1)
    for i ∈ reverse(ψ.sites)
        B = ψ[i]
        @matmul M[x, (σ, y)] := sum(α) B[x, σ, α] * R[α, y]
        R, Q = rq_fact(M, Dcut, tolS, args...)
        R ./= maximum(abs.(R))
        @cast B[x, σ, y] := Q[x, (σ, y)] (σ ∈ 1:size(B, 2))
        ψ[i] = B
    end
end
