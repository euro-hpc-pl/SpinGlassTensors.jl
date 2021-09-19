export rq


function LinearAlgebra.qr(M::AbstractMatrix, Dcut::Int, args...)
    F = pqrfact(M, rank=Dcut, args...)
    q, r, p = F[:Q], F[:R], F[:p]
    _qr_fix(q, r)
end


function rq(M::AbstractMatrix, Dcut::Int, args...)
    F = pqrfact(:c, M, rank=Dcut, args...) 
    q, r = _qr_fix(F[:Q], F[:R])
    r', q'
end


function _qr_fix(Q::T, R::AbstractMatrix) where {T <: AbstractMatrix}
    d = diag(R)
    for i ∈ eachindex(d)
        @inbounds d[i] = ifelse(isapprox(d[i], 0, atol=1e-14), 1, d[i])
    end
    ph = d ./ abs.(d)
    idim = size(R, 1)
    q = T.name.wrapper(Q)[:, 1:idim]
    r = T.name.wrapper(R)[1:idim, :]
    q * Diagonal(ph), Diagonal(ph) * r
end

#=
function LinearAlgebra.svd(A::AbstractMatrix, Dcut::Int, args...)
    U, Σ, V = psvd(A, rank=Dcut, args...)
    d = diag(U)
    for i ∈ eachindex(d)
        @inbounds d[i] = ifelse(isapprox(d[i], 0, atol=1e-14), 1, d[i])
    end
    ph = d ./ abs.(d)
    return  U * Diagonal(ph), Σ, V * Diagonal(ph)
end
=#

function LinearAlgebra.svd(A::AbstractMatrix, Dcut::Int, args...)
    U, Σ, V = svd(A)
    δ = min(Dcut, size(Σ)...)
    U = U[:, 1:δ]
    Σ = Σ[1:δ] 
    V = V[:, 1:δ]

    d = diag(U)
    for i ∈ eachindex(d)
        @inbounds d[i] = ifelse(isapprox(d[i], 0, atol=1e-14), 1, d[i])
    end
    ph = d ./ abs.(d)
    U * Diagonal(ph), Σ, V * Diagonal(ph)
end