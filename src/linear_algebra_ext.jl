export
    rq_fact,
    qr_fact

@inline function phase(d::T; atol::T=eps()) where T <: Real
    isapprox(d, zero(T), atol=atol) ? one(T) : d / abs(d)
end

function LinearAlgebra.svd(A::AbstractMatrix{T}, Dcut::Int=typemax(Int), tol::T=eps(), args...) where T <: Real
    U, Σ, V = svd(A, args...)
    δ = min(Dcut, sum(Σ .> Σ[1] * max(eps(), tol)))
    U, Σ, V = U[:, 1:δ], Σ[1:δ], V[:, 1:δ]
    Σ ./= sum(Σ .^ 2)
    ϕ = reshape((phase.(diag(U); atol=tol)), 1, :)
    U .* ϕ, Σ, V .* ϕ
end


function LowRankApprox.psvd(A::AbstractMatrix{T}, Dcut::Int=typemax(Int), tol::T=eps(), args...) where T <: Real
    U, Σ, V = psvd(A, rank=Dcut, args...)
    Σ ./= sum(Σ .^ 2)
    ϕ = reshape((phase.(diag(U); atol=tol)), 1, :)
    U .* ϕ, Σ, V .* ϕ
end


function qr_fact(M::AbstractMatrix{T}, Dcut::Int=typemax(Int), tol::T=eps(), args...) where T <: Real
    q, r = qr_fix(qr(M, args...))
    Dcut >= size(q, 2) && return q, r
    U, Σ, V = svd(r, Dcut, tol)
    q * U, Σ .* V'
end

function rq_fact(M::AbstractMatrix{T}, Dcut::Int=typemax(Int), tol::T=eps(), args...) where T <: Real
    q, r = qr_fact(M', Dcut, tol, args...)
    r', q'
end

function qr_fix(QR_fact::LinearAlgebra.QRCompactWY; tol::T=eps()) where T <: Real
    ϕ = phase.(diag(QR_fact.R); atol=tol)
    QR_fact.Q * Diagonal(ϕ), ϕ .* QR_fact.R
end
