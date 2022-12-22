module SpinGlassTensors
    using LinearAlgebra, MKL
    using TensorOperations, TensorCast
    using LowRankApprox
    using CUDA, CUDA.CUSPARSE
    using NNlib, NNlibCUDA
    using Memoize

    include("base.jl")
    include("linear_algebra_ext.jl")
    include("mps.jl")
    include("environment.jl")
    include("utils.jl")
    include("canonise.jl")
    include("variational.jl")
    include("gauges.jl")
    include("contractions/dense.jl")
    include("contractions/cuda_sparse.jl")
    include("contractions/central.jl")
    include("contractions/diagonal.jl")
    include("contractions/site.jl")
    include("contractions/virtual.jl")

end # module
