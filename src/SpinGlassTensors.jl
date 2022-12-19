module SpinGlassTensors
    using LinearAlgebra, MKL
    using TensorOperations, TensorCast
    using CUDA, CUDA.CUSPARSE
    using NNlib, NNlibCUDA
    using Memoize

    include("base.jl")
    include("mps.jl")
    include("linear_algebra_ext.jl")
    include("canonise.jl")
    include("compressions.jl")
    include("contractions.jl")
    include("dense.jl")
    include("gauges.jl")
    include("sparse_tensors/cuda_sparse.jl")
    include("sparse_tensors/central.jl")
    include("sparse_tensors/diagonal.jl")
    include("sparse_tensors/site.jl")
    include("sparse_tensors/virtual.jl")

end # module
