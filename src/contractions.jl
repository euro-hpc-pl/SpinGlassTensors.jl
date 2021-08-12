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
    C[]
end

"""
Creates left environment
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

@memoize Dict function left_env(ϕ::AbstractMPS, σ::Vector{Int})
    l = length(σ)
    if l == 0
        L = [1.]
    else
        m = σ[l]
        L̃ = left_env(ϕ, σ[1:l-1])
        M = ϕ[l]
        @reduce L[x] := sum(α) L̃[α] * M[α, $m, x]
    end
    L
end

# NOT tested yet
function right_env(ϕ::AbstractMPS, ψ::AbstractMPS)
    L = length(ψ)
    T = promote_type(eltype(ψ), eltype(ϕ))
    S = typeof(similar(ψ[1], T, (1, 1)))
    R = Vector{S}(undef, L+1)
    R[end] = similar(ψ[1], T, (1, 1))
    R[end][1, 1] = one(T)

    for i ∈ L:-1:1
        M = ψ[i]
        M̃ = conj.(ϕ[i])

        D = R[i+1]
        @tensor D[x, y] := M[x, σ, α] * D[α, β] * M̃[y, σ, β] order = (β, α, σ)
        R[i] = D
    end
    R
end

@memoize Dict function right_env(ϕ::AbstractMPS{T}, W::AbstractMPO{T}, σ::Union{Vector, NTuple}) where {T}
    l = length(σ)
    k = length(W)
    if l == 0
        R = similar(ϕ[1], T, (1, 1))
        R[1, 1] = one(T)
    else
        m = σ[1]
        R̃ = right_env(ϕ, W, σ[2:l])
        M = ϕ[k-l+1]
        M̃ = W[k-l+1]
        @reduce R[x, y] := sum(α, β, γ) M̃[y, $m, β, γ] * M[x, γ, α] * R̃[α, β]
    end
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
function LinearAlgebra.dot(ϕ::AbstractMPS, O::Union{Vector, NTuple}, ψ::AbstractMPS) #where T <: AbstractMatrix
    S = promote_type(eltype(ψ), eltype(ϕ), eltype(O[1]))
    C = similar(ψ[1], S, (1, 1))
    C[1, 1] = one(S)

    for (A, W, B) ∈ zip(ϕ, O, ψ)
        @tensor C[x, y] := conj(A)[β, σ, x] * W[σ, η] * C[β, α] * B[α, η, y] order = (α, η, β, σ)
    end
    C[]
end


function LinearAlgebra.dot(O::AbstractMPO, ψ::AbstractMPS)
    S = promote_type(eltype(ψ), eltype(O))
    T = typeof(ψ)
    ϕ = T.name.wrapper(S, length(ψ))
    for (i, (A, B)) ∈ enumerate(zip(O, ψ))
        @reduce N[(x, a), σ, (y, b)] := sum(η) A[x, σ, y, η] * B[a, η, b]
        ϕ[i] = N
    end
    ϕ
end


function dot!(ψ::AbstractMPS, O::AbstractMPO)
    for (i, (A, B)) ∈ enumerate(zip(ψ, O))
        @reduce N[(x, a), σ, (y, b)] := sum(η) B[x, σ, y, η] * A[a, η, b]
        ψ[i] = N
    end
end


function LinearAlgebra.dot(O1::AbstractMPO, O2::AbstractMPO)
    S = promote_type(eltype(O1), eltype(O2))
    T = typeof(O1)
    O = T.name.wrapper(S, length(O1))
    for (i, (A, B)) ∈ enumerate(zip(O1, O2))
        @reduce V[(x, a), σ, (y, b), η] := sum(γ) A[x, σ, y, γ] * B[a, γ, b, η]
        O[i] = V
    end
    O
end

Base.:(*)(A::AbstractTensorNetwork, B::AbstractTensorNetwork) = dot(A, B)
