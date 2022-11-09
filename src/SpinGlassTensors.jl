module SpinGlassTensors
    using LinearAlgebra, MKL
    using TensorOperations, TensorCast
    #using LowRankApprox
    using CUDA
    using NNlib # This to test batched_multiply
    using NNlibCUDA
    using Memoize

    using DocStringExtensions

    include("base.jl")
    include("linear_algebra_ext.jl")
    include("compressions.jl")
    include("contractions.jl")
    include("dense.jl")
    include("gauges.jl")
    include("sparse_central_tensor.jl")
    include("sparse_diagonal_tensor.jl")
    include("sparse_site_tensor.jl")
    include("sparse_virtual_tensor.jl")

end # module
