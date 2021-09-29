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


# Jezu - K%$urwa max
function LinearAlgebra.dot(O::Mpo, ϕ::Mps, j::Int)
    D = Dict()
    for i ∈ ϕ.sites
        A = O.tensors[i]
        B = ϕ.tensors[i]
        G1 = A[j-4//6]
        V = A[j-1//2]
        T = A[j]
        G2 = A[j+1//6]
        @tensor C[l, x, r, p] := G1[x, y] * V[y, z] * T[l, z, r, d] * G2[d, p] order = (y, z, d)
        @matmul E[(l, x), u, (r, y)] := sum(d) C[l, u, r, d] * B[x, d, y]
        push!(D, i => E)
    end
    Mps(D)
end