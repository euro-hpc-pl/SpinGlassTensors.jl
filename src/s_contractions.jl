export dot, truncate!, contract_left


function LinearAlgebra.dot(ψ::Mps, ϕ::Mps)
    T = promote_type(eltype(ψ.tensors[1]), eltype(ϕ.tensors[1]))
    C = ones(T, 1, 1)

    for (i, (A, B)) ∈ enumerate(zip(ψ, ϕ))
        @tensor C[x, y] := conj(B)[β, σ, x] * C[β, α] * A[α, σ, y] order = (α, β, σ)
    end
    tr(C)
end


LinearAlgebra.norm(ψ::Mps) = sqrt(abs(dot(ψ, ψ)))

function LinearAlgebra.dot(ψ::Mpo, ϕ::Mps)
end


function LinearAlgebra.dot(ψ::AbstractMpo, ϕ::Mps)
end

function contract_left(A,B)
    @cast C[(x,y), u, r] := sum(σ) A[x,σ] * B[(σ,y),u,r] (σ ∈ 1:size(B,2))
    C
end

function contract_up_small(A,B)
    @tensor C[x,y,z] := A[y,σ] * B[x, σ, z] 
    C
end

function contract_up_big(A,B)
    @cast C[(x,y),z,(b, a)] := sum(σ) A[y,z,a,σ] * B[x,σ,b]
    C
end
