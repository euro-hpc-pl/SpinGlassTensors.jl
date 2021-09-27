export dot, truncate!

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

function LinearAlgebra.dot(O::Mpo, ϕ::Mps)
    D = Dict()

    for i ∈ ϕ.sites
        if i == 0 return end
        A = O.tensors[i]
        B = ϕ.tensors[i]
        @matmul C[(x, a), σ, (y, b)] := sum(η) A[x, σ, y, η] * B[a, η, b]
        #@tensor C[l, x, r] := B[l, y, r] * A[y, x]
        push!(D, i => C)
    end
    Mps(D)
end