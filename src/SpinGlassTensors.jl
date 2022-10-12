module SpinGlassTensors
    using LinearAlgebra, MKL
    using TensorOperations, TensorCast
    #using LowRankApprox
    using CUDA
    using NNlib # This to test batched_multiply
    using NNlibCUDA
    #using LoopVectorization # This is to test sparse tensors
    using Memoize

    using DocStringExtensions

    include("base.jl")
    include("linear_algebra_ext.jl")
    include("compressions.jl")
    include("contractions.jl")
end # module
