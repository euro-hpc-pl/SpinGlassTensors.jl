export left_env, right_env, dot!

# --------------------------- Conventions -----------------------
#
#      MPS          MPS*         MPO       left env     right env
#       2            2            2           - 1          2 -
#   1 - A - 3    1 - B - 3    1 - W - 3      L               R
#                                 4           - 2          1 -
# ---------------------------------------------------------------
#
function LinearAlgebra.dot(ϕ::AbstractMPS, ψ::AbstractMPS)
    T = promote_type(eltype(ψ), eltype(ϕ))
    C = ones(T, 1, 1)
    for (A, B) ∈ zip(ψ, ϕ)
        @tensor C[x, y] := conj(B)[β, σ, x] * C[β, α] * A[α, σ, y] order = (α, β, σ)
    end
    tr(C)
end

"""
Creates left environment (ϕ - bra, ψ - ket)
"""
function left_env(ϕ::AbstractMPS, ψ::AbstractMPS)
    T = promote_type(eltype(ψ), eltype(ϕ))
    S = typeof(similar(ψ[1], T, (1, 1)))
    L = Vector{S}(undef, length(ψ)+1)
    L[1] = similar(ψ[1], T, (1, 1))
    L[1][1, 1] = one(T)
    for (i, (A, B)) ∈ enumerate(zip(ψ, ϕ))
        C = L[i]
        @tensor C[x, y] := conj(B)[β, σ, x] * C[β, α] * A[α, σ, y] order = (α, β, σ)
        L[i+1] = C
    end
    L
end

# TODO: remove it (after SpinGlassEngine is updated)
@memoize Dict function left_env(ϕ::AbstractMPS, σ::Vector{Int})
    l = length(σ)
    if l == 0 return ones(eltype(ϕ), 1) end
    m = σ[l]
    L̃ = left_env(ϕ, σ[1:l-1])
    M = ϕ[l]
    @matmul L[x] := sum(α) L̃[α] * M[α, $m, x]
    L
end

"""
Creates right environment (ϕ - bra, ψ - ket)
"""
function right_env(ϕ::AbstractMPS, ψ::AbstractMPS)
    L = length(ψ)
    T = promote_type(eltype(ψ), eltype(ϕ))
    S = typeof(similar(ψ[1], T, (1, 1)))
    R = Vector{S}(undef, L+1)
    R[end] = similar(ψ[1], T, (1, 1))
    R[end][1, 1] = one(T)
    for i ∈ L:-1:1
        M = ψ[i]
        M̃ = ϕ[i]
        D = R[i+1]
        @tensor D[x, y] := M[x, σ, α] * D[α, β] * conj(M̃)[y, σ, β] order = (β, α, σ)
        R[i] = D
    end
    R
end

# TODO: remove it (after SpinGlassEngine is updated)
@memoize Dict function right_env(
    ϕ::AbstractMPS{T}, W::AbstractMPO{T}, σ::Union{Vector, NTuple}
) where {T}
    l = length(σ)
    if l == 0
        R = similar(ϕ[1], T, (1, 1))
        R[1, 1] = one(T)
        return R
    end
    k = length(W)
    R̃ = right_env(ϕ, W, σ[2:l])
    M = ϕ[k-l+1]
    M̃ = W[k-l+1]
    K = @view M̃[:, σ[1], :, :]
    @tensor R[x, y] := K[y, β, γ] * M[x, γ, α] * R̃[α, β] order = (β, γ, α)
    R
end


"""
$(TYPEDSIGNATURES)

Calculates the norm of an MPS \$\\ket{\\phi}\$
"""
LinearAlgebra.norm(ψ::AbstractMPS) = sqrt(abs(dot(ψ, ψ)))


"""
$(TYPEDSIGNATURES)

Calculates \$\\bra{\\phi} O \\ket{\\psi}\$

# Details

Calculates the matrix element of \$O\$
```math
\\bra{\\phi} O \\ket{\\psi}
```
in one pass, utlizing `TensorOperations`.
"""
function LinearAlgebra.dot(ϕ::AbstractMPS, O::Union{Vector, NTuple}, ψ::AbstractMPS)
    S = promote_type(eltype(ψ), eltype(ϕ), eltype(O[1]))
    C = similar(ψ[1], S, (1, 1))
    C[1, 1] = one(S)
    for (A, W, B) ∈ zip(ϕ, O, ψ)
        @tensor C[x, y] := conj(A)[β, σ, x] * W[σ, η] * C[β, α] * B[α, η, y] order = (α, η, β, σ)
    end
    tr(C)
end

function LinearAlgebra.dot(O::AbstractMPO, ψ::AbstractMPS)
    S = promote_type(eltype(ψ), eltype(O))
    T = typeof(ψ)
    ϕ = T.name.wrapper(S, length(ψ))
    for (i, (A, B)) ∈ enumerate(zip(O, ψ))
        @matmul N[(x, a), σ, (y, b)] := sum(η) A[x, σ, y, η] * B[a, η, b]
        ϕ[i] = N
    end
    ϕ
end

function LinearAlgebra.dot(O1::AbstractMPO, O2::AbstractMPO)
    S = promote_type(eltype(O1), eltype(O2))
    T = typeof(O1)
    O = T.name.wrapper(S, length(O1))
    for (i, (A, B)) ∈ enumerate(zip(O1, O2))
        @matmul V[(x, a), σ, (y, b), η] := sum(γ) A[x, σ, y, γ] * B[a, γ, b, η]
        O[i] = V
    end
    O
end
Base.:(*)(A::AbstractTensorNetwork, B::AbstractTensorNetwork) = dot(A, B)
