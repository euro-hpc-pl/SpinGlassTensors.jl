using SpinGlassTensors
using TensorOperations
using TensorCast
using Logging
using LinearAlgebra

#disable_logging(LogLevel(1))

using Test

idx(σ::Int) = (σ == -1) ? 1 : σ + 1


my_tests = [
#    "base.jl",
#    "memoization.jl",
#    "contractions.jl",
#    "compressions.jl",
#    "s_compressions.jl",
    "s_contractions.jl",
#    "identities.jl"
]

for my_test in my_tests
    include(my_test)
end
