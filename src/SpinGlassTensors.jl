module SpinGlassTensors
    using LinearAlgebra, MKL
    using TensorOperations, TensorCast
    using CUDA, CUDA.CUSPARSE
    using NNlib, NNlibCUDA
    using Memoize

    include("base.jl")
    include("linear_algebra_ext.jl")
    include("compressions.jl")
    include("contractions.jl")
    include("dense.jl")
    include("gauges.jl")
    include("sparse_tensors/central.jl")
    include("sparse_tensors/diagonal.jl")
    include("sparse_tensors/site.jl")
    include("sparse_tensors/virtual.jl")

end # module
