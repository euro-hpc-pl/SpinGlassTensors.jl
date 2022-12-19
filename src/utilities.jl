

@inline Base.size(a::AbstractTensorNetwork) = (length(a.tensors), )
@inline Base.length(a::AbstractTensorNetwork) = length(a.tensors)
@inline LinearAlgebra.rank(ψ::QMps) = Tuple(size(A, 2) for A ∈ values(ψ.tensors))
@inline bond_dimension(ψ::QMps) = maximum(size.(values(ψ.tensors), 3))
@inline bond_dimensions(ψ::QMps) = [[size(ten) for ten in values(ψ.tensors[n])] for n ∈ ψ.sites]
@inline bond_dimensions(ψ::QMpo) = [[size(ten) for ten in values(ψ.tensors[n])] for n ∈ ψ.sites]
@inline Base.copy(ψ::QMps) = QMps(copy(ψ.tensors))
@inline Base.:(≈)(a::QMps, b::QMps) = isapprox(a.tensors, b.tensors)
@inline Base.:(≈)(a::QMpo, b::QMpo) = all([isapprox(a.tensors[i], b.tensors[i]) for i ∈ keys(a.tensors)])
