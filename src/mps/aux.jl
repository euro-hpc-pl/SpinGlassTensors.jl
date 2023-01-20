# ./mps/aux.jl: This file provides auxiliary functions to verify various MPS properties.

export
    bond_dimension,
    bond_dimensions,
    is_consistent,
    is_left_normalized,
    is_right_normalized,
    length,
    size

@inline bond_dimension(ψ::QMpsOrMpo) = maximum(size.(values(ψ.tensors), 1))
@inline bond_dimensions(ψ::QMpsOrMpo) = [size(ψ.tensors[n]) for n ∈ ψ.sites]
@inline Base.length(ψ::QMpsOrMpo) = maximum(ψ.sites)
@inline Base.size(ψ::QMpsOrMpo) = (maximum(ψ.sites),)

function is_consistent(ψ::QMps)
    site_min = minimum(ψ.sites)
    site_max = maximum(ψ.sites)
    @assert size(ψ.tensors[site_min], 1) == 1 "Incorrect size on the left boundary."
    @assert size(ψ.tensors[site_max], 2) == 1 "Incorrect size on the right boundary."
    for (s1, s2) ∈ zip(ψ.sites[begin:end-1], ψ.sites[begin+1:end])
        @assert size(ψ.tensors[s1], 2) == size(ψ.tensors[s2], 1) "Incorrect link between $i and $(i+1)."
    end
    a =  ψ.onGPU ? Set((:GPU, )) : Set((:CPU, ))
    println(a)
    b = which_device(ψ)
    println(b)
    @assert a == b
    true
end

eye(ψ, dim) = ψ.onGPU ? CuArray(Diagonal(ones(eltype(ψ), dim))) : Diagonal(ones(eltype(ψ), dim))

function is_left_normalized(ψ::QMps) #TODO rewrite this function
    CUDA.allowscalar(false)
    all(eye(ψ, size(A, 2)) ≈ @tensor Id[x, y] := A[α, x, σ] * A[α, y, σ] order = (α, σ) for A ∈ values(ψ.tensors))
end

function is_right_normalized(ψ::QMps) #TODO rewrite this function
    CUDA.allowscalar(false)
    tmp = []
    for B ∈ values(ψ.tensors)
        @tensor Id[x, y] := B[x, α, σ] * B[y, α, σ] order = (α, σ)
        i2 = eye(ψ, size(B, 1))
        res = i2 ≈ Id
        push!(tmp, res)
    end

    all(tmp)
end
