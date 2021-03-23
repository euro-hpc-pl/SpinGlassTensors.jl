export rq

"""
    LinearAlgebra.qr(M::AbstractMatrix, Dcut::Int, args...)
Wraper to QR.
"""
function LinearAlgebra.qr(M::AbstractMatrix, Dcut::Int, args...)
    fact = pqrfact(M, rank=Dcut, args...)
    Q = fact[:Q]
    R = fact[:R]
    return _qr_fix(Q, R)
end

"""
    rq(M::AbstractMatrix, Dcut::Int, args...)
Wraper to QR.
"""
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

"""
    LinearAlgebra.svd(A::AbstractMatrix, Dcut::Int, args...)
Wraper to SVD.
"""
function LinearAlgebra.svd(A::AbstractMatrix, Dcut::Int, args...)
    U, Σ, V = psvd(A, rank=Dcut, args...)
    d = diag(U)
    ph = d ./ abs.(d)
    return  U * Diagonal(ph), Σ, V * Diagonal(ph)
end
