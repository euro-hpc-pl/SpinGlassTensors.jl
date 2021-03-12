using SpinGlassTensors
using TensorOperations
using TensorCast
using Logging
using LinearAlgebra

disable_logging(LogLevel(1))

using Test

idx(σ::Int) = (σ == -1) ? 1 : σ + 1

my_tests = []
# if CUDA.functional() && CUDA.has_cutensor() && false
#     CUDA.allowscalar(false)
#     include("cuda/test_helpers.jl")
#     push!(
#         my_tests,
#         "cuda/base.jl",
#         "cuda/contractions.jl",
#         "cuda/compressions.jl",
#         "cuda/spectrum.jl"
#     )
# end

my_tests = []

push!(my_tests,
      "base.jl",
      "contractions.jl",
      "compressions.jl",
      "identities.jl",
)

for my_test in my_tests
    include(my_test)
end
