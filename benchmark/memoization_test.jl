using SpinGlassTensors
using Memoization
using CUDA

@memoize Dict function example_cuda_array(::Type{R}, size::Int64) where R <: Real
    CUDA.rand(R, (size, size))
end

@memoize Dict function example_array(::Type{R}, size::Int64) where R <: Real
    rand(R, size, size)
end

A = example_cuda_array(Float64, 1000)
B = example_cuda_array(Float64, 1100)
C = example_array(Float64, 1000)
println(measure_memory(Memoization.caches))


