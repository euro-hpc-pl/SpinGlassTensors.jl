export
    rq_fact,
    qr_fact

function qr_fact(M::AbstractMatrix, Dcut::Int=typemax(Int), tol::Float64=1E-16, args...)
    q, r = _qr_fix(qr(M, args...))
    if Dcut > size(q, 2) return q, r end
    U, Σ, V = svd(r, Dcut, tol)
    q * U, Σ .* V'
end

function rq_fact(M::AbstractMatrix, Dcut::Int=typemax(Int), tol::Float64=1E-15, args...)
    q, r = qr_fact(M', Dcut, tol, args...)
    r', q'
end

function _qr_fix(QR_fact::T) where T <: LinearAlgebra.QRCompactWY
    d = diag(QR_fact.R)
    L = length(d)
    ph = zeros(L, L)
    for i ∈ 1:L
        @inbounds ph[i, i] = ifelse(
            isapprox(d[i], 0, atol=1e-14), 1, d[i] / abs(d[i])
        )
    end
    QR_fact.Q * ph, diag(ph) .* QR_fact.R
end

function LinearAlgebra.svd(
    A::AbstractMatrix, Dcut::Int=typemax(Int), tol::Float64=1E-15, args...
)
    U, Σ, V = svd(A, args...)

    tol = Σ[begin] * max(eps(), tol)
    δ = min(Dcut, sum(Σ .> tol))

    U = U[:, begin:δ]
    Σ = Σ[begin:δ]
    #Σ ./= sum(Σ .^ 2)
    V = V[:, begin:δ]

    d = diag(U)
    for i ∈ eachindex(d)
        @inbounds d[i] = ifelse(isapprox(d[i], 0, atol=1e-14), 1, d[i])
    end
    ph = d ./ abs.(d)
    U * Diagonal(ph), Σ, V * Diagonal(ph)
end
