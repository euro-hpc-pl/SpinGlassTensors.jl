export
    rq_fact,
    qr_fact

@inline phase(d::T; atol=eps()) where T <: Real = isapprox(d, zero(T), atol=atol) ? one(T) : d / abs(d)
@inline phase(d::AbstractArray; atol=eps()) = map(x -> phase(x; atol=atol), d)

function LinearAlgebra.svd(A::AbstractMatrix{T}, Dcut::Int=typemax(Int), tol::T=eps(); kwargs...) where T <: Real
    A = Array(A) # svd is slow on GPU
    U, Σ, V = svd(A; kwargs...)
    δ = min(Dcut, sum(Σ .> Σ[1] * max(eps(), tol)))
    U, Σ, V = U[:, 1:δ], Σ[1:δ], V[:, 1:δ]
    Σ ./= sum(Σ .^ 2)
    ϕ = reshape((phase.(diag(U); atol=tol)), 1, :)
    CuArray.((U .* ϕ, Σ, V .* ϕ))
end

# QR done on CPU
function qr_fact(M::AbstractMatrix{T}, Dcut::Int=typemax(Int), tol::T=eps(); kwargs...) where T <: Real
    M = Array(M)
    q, r = qr_fix(qr(M; kwargs...))
    q, r = CuArray(q), CuArray(r)
    Dcut >= size(q, 2) && return q, r
    U, Σ, V = svd(r, Dcut, tol)
    CuArray(q * U), CuArray(Σ .* V')
end

#=
# QR done on GPU
function qr_fact(M::AbstractMatrix{T}, Dcut::Int=typemax(Int), tol::T=eps(); kwargs...) where T <: Real
    CUDA.allowscalar(true)
    q, r = qr_fix(qr(M; kwargs...))
    CUDA.allowscalar(false)
    Dcut >= size(q, 2) && return q, r
    U, Σ, V = svd(r, Dcut, tol)
    q * U, Σ .* V'
end
=#

function rq_fact(M::AbstractMatrix{T}, Dcut::Int=typemax(Int), tol::T=eps(); kwargs...) where T <: Real
    q, r = qr_fact(M', Dcut, tol; kwargs...)
    CuArray.((r', q'))
end

function qr_fix(QR_fact; tol::T=eps()) where T <: Real
    ϕ = phase(diag(QR_fact.R); atol=tol)
    QR_fact.Q * Diagonal(ϕ), ϕ .* QR_fact.R
#   QR_fact.Q * CuArray(Diagonal(ϕ)), ϕ .* QR_fact.R # TODO can we get rid of CuArray?
end
