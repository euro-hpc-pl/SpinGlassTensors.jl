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

    # include("base.jl")
    include("s_base.jl")
    include("linear_algebra_ext.jl")
    include("compressions.jl")
    # include("custom_cuda.jl")
    include("s_compressions.jl")
    # include("identities.jl")
    # include("contractions.jl")
    include("s_contractions.jl")
end # module
