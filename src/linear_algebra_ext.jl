export rq

function LinearAlgebra.qr(M::AbstractMatrix, Dcut::Int, args...)
    fact = pqrfact(M, rank=Dcut, args...)
    Q = fact[:Q]
    R = fact[:R]
    return _qr_fix(Q, R)
end

function rq(M::AbstractMatrix, Dcut::Int, args...)
    fact = pqrfact(:c, conj.(M), rank=Dcut, args...)
    Q = fact[:Q]
    R = fact[:R]
    return _qr_fix(Q, R)'
end

function _qr_fix(Q::T, R::AbstractMatrix) where {T <: AbstractMatrix}
    d = diag(R)
    ph = d./abs.(d)
    idim = size(R, 1)
    q = T.name.wrapper(Q)[:, 1:idim]
    return transpose(ph) .* q
end

function LinearAlgebra.svd(A::AbstractMatrix, Dcut::Int, args...)
    U, Σ, V = psvd(A, rank=Dcut, args...)
    d = diag(U)
    ph = d ./ abs.(d)
    for i ∈ eachindex(ph)
        @inbounds ph[i] = ifelse(isapprox(ph[i], 0, atol=1e-14), 1, ph[i])
    end
    return  U * Diagonal(ph), Σ, V * Diagonal(ph)
end