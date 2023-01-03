export
    rq_fact,
    qr_fact

@inline phase(d::T; atol::T=eps()) where T <: Real = isapprox(d, zero(T), atol=atol) ? one(T) : d / abs(d)

function LinearAlgebra.svd(A::AbstractMatrix{T}, Dcut::Int=typemax(Int), tol::T=eps(); kwargs...) where T <: Real
    A = Array(A) # svd is slow on GPU
    U, Σ, V = svd(A; kwargs...)
    δ = min(Dcut, sum(Σ .> Σ[1] * max(eps(), tol)))
    U, Σ, V = U[:, 1:δ], Σ[1:δ], V[:, 1:δ]
    Σ ./= sum(Σ .^ 2)
    ϕ = reshape((phase.(diag(U); atol=tol)), 1, :)
    CuArray.((U .* ϕ, Σ, V .* ϕ))
end


function qr_fact(M::AbstractMatrix{T}, Dcut::Int=typemax(Int), tol::T=eps(); kwargs...) where T <: Real
    M = Array(M) # TODO to be fixed
    q, r = qr_fix(qr(M; kwargs...))
    q, r = CuArray(q), CuArray(r)
    Dcut >= size(q, 2) && return q, r
    U, Σ, V = svd(r, Dcut, tol)
    CuArray(q * U), CuArray(Σ .* V')
end

#=
function qr_fact(M::AbstractMatrix{T}, Dcut::Int=typemax(Int), tol::T=eps(); kwargs...) where T <: Real
#    q, r = qr_fix(qr(M; kwargs...))
    q, r = qr(M; kwargs...)
    Dcut >= size(q, 2) && return q, r
    U, Σ, V = svd(r, Dcut, tol)
    q * U, Σ .* V'
end
=#

function rq_fact(M::AbstractMatrix{T}, Dcut::Int=typemax(Int), tol::T=eps(); kwargs...) where T <: Real
    q, r = qr_fact(M', Dcut, tol; kwargs...)
    CuArray.((r', q'))
    #r', q'
end

function qr_fix(QR_fact; tol::T=eps()) where T <: Real  #LinearAlgebra.QRCompactWY
    ϕ = phase.(diag(QR_fact.R); atol=tol)
    QR_fact.Q * Diagonal(ϕ), ϕ .* QR_fact.R
end
