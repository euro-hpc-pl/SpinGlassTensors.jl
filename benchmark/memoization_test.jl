using SpinGlassTensors
using Memoization
using CUDA


@memoize Dict function example_cuda_array(::Type{R}, size::Int64) where R <: Real
    CUDA.rand(R, (size, size))
end


@memoize Dict function example_array(::Type{R}, size::Int64) where R <: Real
    rand(R, size, size)
end


@memoize Dict function aux_cusparse(::Type{R}, n::Int64) where R <: Real
    CuArray(1:n+1), CUDA.ones(R, n)
end


@memoize Dict function CUDA.CUSPARSE.CuSparseMatrixCSC(::Type{R}, p::Vector{Int}) where R <: Real
    n = length(p)
    cn, co = aux_cusparse(R, n)
    CUDA.CUSPARSE.CuSparseMatrixCSC(cn, CuArray(p), co, (maximum(p), n))
end


A = example_cuda_array(Float64, 10000)
B = example_cuda_array(Float64, 1100)
C = example_array(Float64, 1000)
p = rand(1:5000, 100000000)
D = CUDA.CUSPARSE.CuSparseMatrixCSC(Float64, p)
CUDA.memory_status()
println("/n")
measure_memory(Memoization.caches)



