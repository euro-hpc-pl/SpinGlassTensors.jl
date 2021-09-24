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

"""
$(TYPEDSIGNATURES)

Calculates the norm of an MPS \$\\ket{\\phi}\$
"""
LinearAlgebra.norm(ψ::Mps) = sqrt(abs(dot(ψ, ψ)))


function LinearAlgebra.dot(O::Mpo, ψ::Mps)
    
end


function LinearAlgebra.dot(O1::Mpo, O2::Mpo)
    W = Dict()
    for i ∈ O1.sites
        A, B = O1.tensors[i], O2.tensors[i]
        @matmul V[(x, a), σ, (y, b), η] := sum(γ) A[x, σ, y, γ] * B[a, γ, b, η]
        push!(W, i => V)
    end
    Mpo(W)
end 