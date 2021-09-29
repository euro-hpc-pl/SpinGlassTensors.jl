export dot, truncate!

function LinearAlgebra.dot(ψ::Mps, ϕ::Mps)
    T = promote_type(eltype(ψ.tensors[1]), eltype(ϕ.tensors[1]))
    C = ones(T, 1, 1)

    for i ∈ ψ.sites
    for (i, (A, B)) ∈ enumerate(zip(ψ, ϕ))
        @tensor C[x, y] := conj(B)[β, σ, x] * C[β, α] * A[α, σ, y] order = (α, β, σ)
    end
    tr(C)
end

LinearAlgebra.norm(ψ::Mps) = sqrt(abs(dot(ψ, ψ)))
