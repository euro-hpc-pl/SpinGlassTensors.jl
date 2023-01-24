# @memoize Dict
function CUDA.CUSPARSE.CuSparseMatrixCSC(::Type{R}, p::CuArray{Int64, 1}) where R <: Real
    n = length(p)
    mp = maximum(p)
    CuSparseMatrixCSC(CuArray(1:n+1), p, CUDA.ones(R, n), (mp, n))
end

function CUDA.CUSPARSE.CuSparseMatrixCSC(::Type{T}, p1::R, p2::R, p3::R) where {T <: Real, R <: CuArray{Int64, 1}}
    @assert length(p1) == length(p2) == length(p3)
    s1, s2 = maximum(p1), maximum(p2)
    p = p1 .+ s1 * (p2 .- 1) .+ s1 * s2 * (p3 .- 1)
    CuSparseMatrixCSC(T, p)
end

