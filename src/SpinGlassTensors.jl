module SpinGlassTensors
    using LinearAlgebra, MKL
    using TensorOperations, TensorCast
    using LowRankApprox, TSVD
    using CUDA, CUDA.CUSPARSE
    using NNlib, NNlibCUDA
    using Memoize

    CUDA.allowscalar(false)

    include("base.jl")
    include("linear_algebra_ext.jl")
    include("mps.jl")
    include("environment.jl")
    include("utils.jl")
    include("canonise.jl")
    include("variational.jl")
    include("zipper.jl")
    include("gauges.jl")
    include("contractions/dense.jl")
    include("contractions/cuda_sparse.jl")
    include("contractions/central.jl")
    include("contractions/diagonal.jl")
    include("contractions/site.jl")
    include("contractions/virtual.jl")
    include("contractions_cuda/dense.jl")
    include("contractions_cuda/site.jl")

end # module
