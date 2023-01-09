# ./mps/aux.jl: This file provides auxiliary functions to verify various MPS properties.

export
    bond_dimension,
    bond_dimensions,
    verify_bonds,
    is_left_normalized,
    is_right_normalized,
    length,
    size

@inline bond_dimension(ψ::QMpsOrMpo) = maximum(size.(values(ψ.tensors), 3))
@inline bond_dimensions(ψ::QMpsOrMpo) = [size(ψ.tensors[n]) for n ∈ ψ.sites]
@inline Base.length(ψ::QMpsOrMpo) = maximum(ψ.sites)
@inline Base.size(ψ::QMpsOrMpo) = (maximum(ψ.sites),)

function verify_bonds(ψ::QMps)
    L = maximum(ψ.sites)
    @assert size(ψ.tensors[1], 1) == 1 "Incorrect size on the left boundary."
    @assert size(ψ.tensors[L], 3) == 1 "Incorrect size on the right boundary."
    for (i, j) ∈ enumerate(ψ.sites[begin:end-1]) 
        @assert size(ψ.tensors[j], 3) == size(ψ.tensors[ψ.sites[i+1]], 1) "Incorrect link between $i and $(i+1)." 
    end
end

function is_left_normalized(ψ::QMps) #TODO rewrite this function
    all(I(size(A, 3)) ≈ @tensor Id[x, y] := conj(A)[α, σ, x] * A[α, σ, y] order = (α, σ) for A ∈ values(ψ.tensors))
end

function is_right_normalized(ϕ::QMps) #TODO rewrite this function
    all(I(size(B, 1)) ≈ @tensor Id[x, y] := B[x, σ, α] * conj(B)[y, σ, α] order = (α, σ) for B ∈ values(ϕ.tensors))
end
