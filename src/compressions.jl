export canonise!, truncate!, compress!, compress

# TODO: check if we need this
# This is for backwards compatibility
"""
$(TYPEDSIGNATURES)

"""
function compress(ϕ::AbstractMPS, Dcut::Int, tol::Number=1e-8, max_sweeps::Int=4)
    ψ = copy(ϕ)
    compress!(ψ, Dcut, tol, max_sweeps)
    ψ
end

"""
$(TYPEDSIGNATURES)

"""
function compress!(ϕ::AbstractMPS, Dcut::Int, tol::Number=1e-8, max_sweeps::Int=4)
    # Right canonise ϕ
    _left_sweep!(ϕ)

    # Initial guess - truncated ϕ
    ψ = copy(ϕ)
    _right_sweep!(ϕ, Dcut)

    # Create environment
    env = left_env(ϕ, ψ)

    # Variational compression
    overlap = Inf
    overlap_before = -Inf

    @info "Compressing state down to" Dcut

    for sweep ∈ 1:max_sweeps
        _left_sweep_var!!(ϕ, env, ψ)
        overlap = _right_sweep_var!!(ϕ, env, ψ)

        diff = abs(overlap_before - abs(overlap))
        @info "Convergence" diff

        if diff < tol
            @info "Finished in $sweep sweeps of $(max_sweeps)."
            return overlap
        else
            overlap_before = overlap
        end
    end
    overlap
end

"""
$(TYPEDSIGNATURES)

"""
function truncate!(ψ::AbstractMPS, s::Symbol, Dcut::Int=typemax(Int))
    @assert s ∈ (:left, :right)
    if s == :right
        _right_sweep!(ψ)
        _left_sweep!(ψ, Dcut)
    else
        _left_sweep!(ψ)
        _right_sweep!(ψ, Dcut)
    end
end

"""
$(TYPEDSIGNATURES)

"""
canonise!(ψ::AbstractMPS, s::Symbol) = canonise!(ψ, Val(s))

"""
$(TYPEDSIGNATURES)

"""
canonise!(ψ::AbstractMPS, ::Val{:right}) = _left_sweep!(ψ, typemax(Int))

"""
$(TYPEDSIGNATURES)

"""
canonise!(ψ::AbstractMPS, ::Val{:left}) = _right_sweep!(ψ, typemax(Int))

"""
$(TYPEDSIGNATURES)

"""
function _right_sweep!(ψ::AbstractMPS, Dcut::Int=typemax(Int))
    R = ones(eltype(ψ), 1, 1)
    for (i, A) ∈ enumerate(ψ)
        @matmul M̃[(x, σ), y] := sum(α) R[x, α] * A[α, σ, y]
        Q, R = qr_fact(M̃, Dcut)
        R = R ./ maximum(abs.(R))
        @cast A[x, σ, y] := Q[(x, σ), y] (σ ∈ 1:size(A, 2))
        ψ[i] = A
    end
end

"""
$(TYPEDSIGNATURES)

"""
function _left_sweep!(ψ::AbstractMPS, Dcut::Int=typemax(Int))
    R = ones(eltype(ψ), 1, 1)
    for i ∈ length(ψ):-1:1
        B = ψ[i]
        @matmul M̃[x, (σ, y)] := sum(α) B[x, σ, α] * R[α, y]
        R, Q = rq_fact(M̃, Dcut)
        R = R ./ maximum(abs.(R))
        @cast B[x, σ, y] := Q[x, (σ, y)] (σ ∈ 1:size(B, 2))
        ψ[i] = B
    end
end

"""
$(TYPEDSIGNATURES)

"""
function _right_sweep(A::AbstractArray, Dcut::Int=typemax(Int))
    rank = ndims(A)
    ψ = MPS(eltype(A), rank)
    R = reshape(copy(A), (1, length(A)))

    for i ∈ 1:rank
        d = size(A, i)
        @cast M[(x, σ), y] := R[x, (σ, y)] (σ ∈ 1:d)
        Q, R = qr_fact(M, Dcut)
        R = R ./ maximum(abs.(R))
        @cast B[x, σ, y] := Q[(x, σ), y] (σ ∈ 1:d)
        ψ[i] = B
    end
    ψ
end

"""
$(TYPEDSIGNATURES)

"""
function _left_sweep(A::AbstractArray, Dcut::Int=typemax(Int))
    rank = ndims(A)
    ψ = MPS(eltype(A), rank)
    R = reshape(copy(A), (length(A), 1))

    for i ∈ rank:-1:1
        d = size(A, i)
        @cast M[x, (σ, y)] := R[(x, σ), y] (σ ∈ 1:d)
        R, Q = rq_fact(M, Dcut)
        R = R ./ maximum(abs.(R))
        @cast B[x, σ, y] := Q[x, (σ, y)] (σ ∈ 1:d)
        ψ[i] = B
    end
    ψ
end

"""
$(TYPEDSIGNATURES)

"""
function _left_sweep_var!!(
    ϕ::AbstractMPS, env::Vector{<:AbstractMatrix}, ψ::AbstractMPS
)
    env[end] = ones(eltype(ϕ), 1, 1)

    for i ∈ length(ψ):-1:1
        L, R = env[i], env[i+1]

        # Optimize site
        M = ψ[i]
        @tensor MM[x, σ, α] := L[x, β] * M[β, σ, α]
        @matmul MM[x, (σ, y)] := sum(α) MM[x, σ, α] * R[α, y]

        _, Q = rq_fact(MM)
        @cast B[x, σ, y] := Q[x, (σ, y)] (σ ∈ 1:size(M, 2))

        # Update ϕ and right environment
        ϕ[i] = B
        A = ψ[i]

        @tensor RR[x, y] := A[x, σ, α] * R[α, β] * conj(B)[y, σ, β] order = (β, α, σ)
        env[i] = RR
    end
    env[1][1]
end

"""
$(TYPEDSIGNATURES)

"""
function _right_sweep_var!!(
    ϕ::AbstractMPS, env::Vector{<:AbstractMatrix}, ψ::AbstractMPS
)
    env[1] = ones(eltype(ϕ), 1, 1)

    for (i, M) ∈ enumerate(ψ)
        L, R = env[i], env[i+1]

        # Optimize site
        @tensor M̃[x, σ, α] := L[x, β] * M[β, σ, α]
        @matmul B[(x, σ), y] := sum(α) M̃[x, σ, α] * R[α, y]

        Q, _ = qr_fact(B)
        @cast A[x, σ, y] := Q[(x, σ), y] (σ ∈ 1:size(M, 2))

        # Update ϕ and left environment
        ϕ[i] = A
        B = ψ[i]

        @tensor LL[x, y] := conj(A[β, σ, x]) * L[β, α] * B[α, σ, y] order = (α, β, σ)
        env[i+1] = LL
    end
    env[end][1]
end
