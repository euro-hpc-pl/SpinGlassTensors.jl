export rq_fact, qr_fact


function qr_fact(M::AbstractMatrix, Dcut::Int=typemax(Int), tol::Float64=1E-12, args...)
    F = qr(M, args...)
    q, r = _qr_fix(Array(F.Q), Array(F.R))
    if Dcut > max(size(M)...) return q, r end
    U, Σ, V = svd(r, Dcut, tol)
    q * U, Diagonal(Σ) * V'
end


function rq_fact(M::AbstractMatrix, Dcut::Int=typemax(Int), tol::Float64=1E-12, args...)
    q, r = qr_fact(M', Dcut, tol, args...) 
    r', q'
end


function _qr_fix(Q::T, R::AbstractMatrix) where {T <: AbstractMatrix}
    d = diag(R)
    for i ∈ eachindex(d)
        @inbounds d[i] = ifelse(isapprox(d[i], 0, atol=1e-14), 1, d[i])
    end
    ph = d ./ abs.(d)
    Q * Diagonal(ph), Diagonal(ph) * R
end


function LinearAlgebra.svd(A::AbstractMatrix, Dcut::Int=typemax(Int), tol::Float64=1E-12, args...)
    #U, Σ, V = psvd(A, rank=Dcut, args...)
 
    U, Σ, V = svd(A, args...)

    tol = Σ[1] * max(eps(), tol)
    δ = min(Dcut, sum(Σ .> tol))

    U = U[:, 1:δ]
    Σ = Σ[1:δ] 
    Σ ./ sum(Σ) 
    V = V[:, 1:δ]

    d = diag(U)
    for i ∈ eachindex(d)
        @inbounds d[i] = ifelse(isapprox(d[i], 0, atol=1e-14), 1, d[i])
    end
    ph = d ./ abs.(d)
    U * Diagonal(ph), Σ, V * Diagonal(ph)
end
