using SpinGlassTensors
using TensorOperations
using TensorCast
using Logging
using LinearAlgebra
using CUDA

disable_logging(LogLevel(1))

using Test

my_tests = [
    "canonise.jl",
    "projectors.jl",
    # "variational.jl"
    ]

for my_test in my_tests
    include(my_test)
end
