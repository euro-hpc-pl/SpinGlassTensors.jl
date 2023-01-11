using SpinGlassTensors
using TensorOperations
using TensorCast
using Logging
using LinearAlgebra

disable_logging(LogLevel(1))

using Test

my_tests = [
    #"mps.jl",
    "canonise.jl",
    #"environment.jl"
    ]


for my_test in my_tests
    include(my_test)
end
