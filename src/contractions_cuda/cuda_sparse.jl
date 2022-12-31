function CUDA.CUSPARSE.CuSparseMatrixCSC(::Type{R}, pr::Vector{Int}) where R <: Real
    n = length(pr)
    CuSparseMatrixCSC(CuArray(1:n+1), CuArray(pr), CUDA.ones(R, n), (maximum(pr), n))
end

function CUDA.CUSPARSE.CuSparseMatrixCSC(::Type{T}, p_lb::R, p_l::R, p_lt::R) where {T <: Real, R <: Vector{Int}}
    @assert length(p_lb) == length(p_l) == length(p_lt)

    p_l, p_lb, p_lt = CuArray.((p_l, p_lb, p_lt))
    ncol = length(p_lb)
    n = maximum(p_l)
    m = maximum(p_lb)

    CuSparseMatrixCSC(
        CuArray(1:ncol+1), n * m * (p_lt .- 1) .+ m * (p_l .- 1) .+ p_lb, CUDA.ones(T, ncol), (n * m * maximum(p_lt), ncol)
    )
end

function CUDA.CUSPARSE.CuSparseMatrixCSR(::Type{T}, p_lb::R, p_l::R, p_lt::R) where {T <: Real, R <: Vector{Int}}
    transpose(CuSparseMatrixCSC(T, p_lb, p_l, p_lt))
end
