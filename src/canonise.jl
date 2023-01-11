
# canonise.jl: This file provides basic function to left / right truncate / canonise MPS. CUDA is supported.

export
    canonise!,
    truncate!,
    canonise_truncate!

function truncate!(ψ::QMps{T}, s::Symbol, Dcut::Int=typemax(Int), tolS::T=eps(); kwargs...) where T <: Real
    @assert s ∈ (:left, :right)
    if s == :right
        _right_sweep!(ψ; kwargs...)
        _left_sweep!(ψ, Dcut, tolS; kwargs...)
    else
        _left_sweep!(ψ, args...)
        _right_sweep!(ψ, Dcut, tolS; kwargs...)
    end
end

canonise!(ψ::QMps, s::Symbol) = canonise!(ψ, Val(s))
canonise!(ψ::QMps, ::Val{:right}) = _left_sweep!(ψ, typemax(Int))
canonise!(ψ::QMps, ::Val{:left}) = _right_sweep!(ψ, typemax(Int))

function canonise_truncate!(ψ::QMps, type::Symbol, Dcut::Int=typemax(Int), tolS=eps(); kwargs...)
    if type == :right
        _left_sweep!(ψ, Dcut, tolS; kwargs...)
    elseif type == :left
        _right_sweep!(ψ, Dcut, tolS; kwargs...)
    else
        throw(ArgumentError("Wrong canonization type $type"))
    end
end

function _right_sweep!(ψ::QMps{T}, Dcut::Int=typemax(Int), tolS::T=eps(); kwargs...) where T <: Real
    R = ψ.onGPU ? CUDA.ones(T, 1, 1) : ones(T, 1, 1)
    for i ∈ ψ.sites
        A = ψ[i]
        @matmul M[(x, σ), y] := sum(α) R[x, α] * A[α, σ, y]
        Q, R = qr_fact(M, Dcut, tolS; toGPU = ψ.onGPU, kwargs...)
        R ./= maximum(abs.(R))
        @cast A[x, σ, y] := Q[(x, σ), y] (σ ∈ 1:size(A, 2))
        ψ[i] = A
    end
end

function _left_sweep!(ψ::QMps{T}, Dcut::Int=typemax(Int), tolS::T=eps(); kwargs...) where T <: Real
    R = ψ.onGPU ? CUDA.ones(T, 1, 1) : ones(T, 1, 1)
    for i ∈ reverse(ψ.sites)
        B = ψ[i]
        @matmul M[x, (σ, y)] := sum(α) B[x, σ, α] * R[α, y]
        R, Q = rq_fact(M, Dcut, tolS; toGPU = ψ.onGPU, kwargs...)
        R ./= maximum(abs.(R))
        @cast B[x, σ, y] := Q[x, (σ, y)] (σ ∈ 1:size(B, 2))
        ψ[i] = B
    end
end
