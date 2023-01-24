@memoize Dict function CUDA.CUSPARSE.CuSparseMatrixCSC(::Type{R}, p::Vector{Int}) where R <: Real
    n = length(p)
    CuSparseMatrixCSC(CuArray(1:n+1), CuArray(p), CUDA.ones(R, n), (maximum(p), n))
end

function CUDA.CUSPARSE.CuSparseMatrixCSC(::Type{T}, p1::R, p2::R, p3::R) where {T <: Real, R <: Vector{Int}}
    @assert length(p1) == length(p2) == length(p3)
    s1, s2 = maximum(p1), maximum(p2)
    p = p1 .+ s1 * (p2 .- 1) .+ s1 * s2 * (p3 .- 1)
    CuSparseMatrixCSC(T, p)
end

