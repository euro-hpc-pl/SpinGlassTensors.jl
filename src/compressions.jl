export canonise!, truncate!, compress!, compress


# This is for backwards compatibility
function compress(
    ϕ::AbstractMPS,
    Dcut::Int,
    tol::Number = 1E-8,
    max_sweeps::Int = 4,
    args...,
)
    ψ = copy(ϕ)
    compress!(ψ, Dcut, tol, max_sweeps, args...)
    ψ
end

"""
    compress!(
        ϕ::AbstractMPS,
        Dcut::Int,
        tol::Number = 1E-8,
        max_sweeps::Int = 4,
        args...,
    )

# Arguments

- `ϕ::AbstractMPS`: the input MPS to be compressed
- `Dcut::Int`: the maximum bond dimension of the compressed MPS
- `tol::Number = 1E-8`: the tolerance threshold for convergence of the iterative compression process (default value: 1E-8)
- `max_sweeps::Int = 4`: the maximum number of iterations allowed for the compression process (default value: 4)

# Output
- `overlap`: The overlap of the compressed MPS with the original input MPS.
"""
function compress!(
    ϕ::AbstractMPS,
    Dcut::Int,
    tol::Number = 1E-8,
    max_sweeps::Int = 4,
    args...,
)
    # right canonise ϕ
    _left_sweep!(ϕ, args...)

    # Initial guess - truncated ϕ
    ψ = copy(ϕ)
    _right_sweep!(ϕ, Dcut, args...)

    # Create environment
    env = left_env(ϕ, ψ)

    # Variational compression
    overlap = Inf
    overlap_before = -Inf

    @info "Compressing state down to" Dcut

    for sweep ∈ 1:max_sweeps
        _left_sweep_var!!(ϕ, env, ψ, args...)
        overlap = _right_sweep_var!!(ϕ, env, ψ, args...)

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
    truncate!(ψ::AbstractMPS, s::Symbol, Dcut::Int = typemax(Int), args...)

Truncate the bond dimension of a matrix product state (MPS) in either 
the left or right canonical form, depending on the value of the `s` input argument.

# Arguments

- `ψ::AbstractMPS`: the input MPS to be truncated
- `s::Symbol`: determines whether to truncate the MPS in the left or right canonical form. Must be one of the following values:
    - `:left`: truncate in left canonical form
    - `:right`: truncate in right canonical form
- `Dcut::Int`: the maximum bond dimension to which the MPS should be truncated. 

"""
function truncate!(ψ::AbstractMPS, s::Symbol, Dcut::Int = typemax(Int), args...)
    @assert s ∈ (:left, :right)
    if s == :right
        _right_sweep!(ψ, args...)
        _left_sweep!(ψ, Dcut, args...)
    else
        _left_sweep!(ψ, args...)
        _right_sweep!(ψ, Dcut, args...)
    end
end

"""
    canonise!(ψ::AbstractMPS, s::Symbol)

canonizes a matrix product state (MPS) in either the left or right canonical form, 
depending on the value of the `s` input argument. Must be one of the following values:
- `:left`: canonize in left canonical form
- `:right`: canonize in right canonical form

"""
canonise!(ψ::AbstractMPS, s::Symbol) = canonise!(ψ, Val(s))

canonise!(ψ::AbstractMPS, ::Val{:right}) = _left_sweep!(ψ, typemax(Int))
canonise!(ψ::AbstractMPS, ::Val{:left}) = _right_sweep!(ψ, typemax(Int))


function _right_sweep!(ψ::AbstractMPS, Dcut::Int = typemax(Int), args...)
    R = ones(eltype(ψ), 1, 1)
    for (i, A) ∈ enumerate(ψ)
        @matmul M̃[(x, σ), y] := sum(α) R[x, α] * A[α, σ, y]
        Q, R = qr_fact(M̃, Dcut, args...)
        R = R ./ maximum(abs.(R))
        @cast A[x, σ, y] := Q[(x, σ), y] (σ ∈ 1:size(A, 2))
        ψ[i] = A
    end
end


function _left_sweep!(ψ::AbstractMPS, Dcut::Int = typemax(Int), args...)
    R = ones(eltype(ψ), 1, 1)
    for i ∈ length(ψ):-1:1
        B = ψ[i]
        @matmul M̃[x, (σ, y)] := sum(α) B[x, σ, α] * R[α, y]
        R, Q = rq_fact(M̃, Dcut, args...)
        R = R ./ maximum(abs.(R))
        @cast B[x, σ, y] := Q[x, (σ, y)] (σ ∈ 1:size(B, 2))
        ψ[i] = B
    end
end


function _left_sweep_var!!(
    ϕ::AbstractMPS,
    env::Vector{<:AbstractMatrix},
    ψ::AbstractMPS,
    args...,
)
    env[end] = ones(eltype(ϕ), 1, 1)

    for i ∈ length(ψ):-1:1
        L, R = env[i], env[i+1]

        # optimize site
        M = ψ[i]
        @tensor MM[x, σ, α] := L[x, β] * M[β, σ, α]
        @matmul MM[x, (σ, y)] := sum(α) MM[x, σ, α] * R[α, y]

        _, Q = rq_fact(MM, args...)
        @cast B[x, σ, y] := Q[x, (σ, y)] (σ ∈ 1:size(M, 2))

        # update ϕ and right environment
        ϕ[i] = B
        A = ψ[i]

        @tensor RR[x, y] := A[x, σ, α] * R[α, β] * conj(B)[y, σ, β] order = (β, α, σ)
        env[i] = RR
    end
    env[1][1]
end


function _right_sweep_var!!(
    ϕ::AbstractMPS,
    env::Vector{<:AbstractMatrix},
    ψ::AbstractMPS,
    args...,
)
    env[1] = ones(eltype(ϕ), 1, 1)

    for (i, M) ∈ enumerate(ψ)
        L, R = env[i], env[i+1]

        # optimize site
        @tensor M̃[x, σ, α] := L[x, β] * M[β, σ, α]
        @matmul B[(x, σ), y] := sum(α) M̃[x, σ, α] * R[α, y]

        Q, _ = qr_fact(B, args...)
        @cast A[x, σ, y] := Q[(x, σ), y] (σ ∈ 1:size(M, 2))

        # update ϕ and left environment
        ϕ[i] = A
        B = ψ[i]

        @tensor LL[x, y] := conj(A[β, σ, x]) * L[β, α] * B[α, σ, y] order = (α, β, σ)
        env[i+1] = LL
    end
    env[end][1]
end


function _right_sweep(
    A::AbstractArray,
    Dcut::Int = typemax(Int),
    args...,
)
    rank = ndims(A)
    ψ = MPS(eltype(A), rank)
    R = reshape(copy(A), (1, length(A)))

    for i ∈ 1:rank
        d = size(A, i)
        @cast M[(x, σ), y] := R[x, (σ, y)] (σ ∈ 1:d)
        Q, R = qr_fact(M, Dcut, args...)
        R = R ./ maximum(abs.(R))
        @cast B[x, σ, y] := Q[(x, σ), y] (σ ∈ 1:d)
        ψ[i] = B
    end
    ψ
end


function _left_sweep(
    A::AbstractArray,
    Dcut::Int = typemax(Int),
    args...,
)
    rank = ndims(A)
    ψ = MPS(eltype(A), rank)
    R = reshape(copy(A), (length(A), 1))

    for i ∈ rank:-1:1
        d = size(A, i)
        @cast M[x, (σ, y)] := R[(x, σ), y] (σ ∈ 1:d)
        R, Q = rq_fact(M, Dcut, args...)
        R = R ./ maximum(abs.(R))
        @cast B[x, σ, y] := Q[x, (σ, y)] (σ ∈ 1:d)
        ψ[i] = B
    end
    ψ
end
