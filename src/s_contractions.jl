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

function contract_left(A::AbstractMatrix{T}, B::AbstractArray{T,3}) where {T}
    @cast C[(x,y), u, r] := sum(σ) A[x,σ] * B[(σ,y),u,r] (σ ∈ 1:size(B,2))
    C
end

function contract_up(A::AbstractMatrix{T}, B::AbstractArray{T,3}) where {T}
    @tensor C[l,u,r] := A[u,σ] * B[l, σ, r] 
    C
end

function contract_up(A::AbstractArray{T,3}, B::AbstractArray{T,4}) where {T}
    @cast C[(x,y),z,(b, a)] := sum(σ) B[y,z,a,σ] * A[x,σ,b] 
    C
end

function contract_up(A::AbstractMatrix{T}, B::AbstractArray{T,4}) where {T}
    @tensor C[l,u,r,d] := A[u,σ] * B[l, σ, r, d] 
    C
end

function contract_down(A::AbstractMatrix{T}, B::AbstractArray{T,4}) where {T}
    @tensor C[l,u,r,d] := B[l,u,r,σ] * A[σ, d]
    C
end