export dot

function LinearAlgebra.dot(ψ::Mps, ϕ::Mps)
    T = promote_type(eltype(ψ.tensors[1]), eltype(ϕ.tensors[1]))
    C = ones(T, 1, 1)

    for i ∈ ψ.sites
        A, B = ψ.tensors[i], ϕ.tensors[i]
        @tensor C[x, y] := conj(B)[β, σ, x] * C[β, α] * A[α, σ, y] order = (α, β, σ)
    end
    tr(C)
end

LinearAlgebra.norm(ψ::Mps) = sqrt(abs(dot(ψ, ψ)))


