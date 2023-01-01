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
    include("./mps/base.jl")
    include("./mps/transpose.jl")
    include("./mps/dot.jl")
    include("./mps/identity.jl")
    include("transfer.jl")
    include("environment.jl")
    include("utils/memory.jl")
    include("canonise.jl")
    include("variational.jl")
    include("zipper.jl")
    include("gauges.jl")
    # TODO this is to be eventually merged / removed
    include("contractions/dense.jl")
    include("contractions/central.jl")
    include("contractions/diagonal.jl")
    include("contractions/site.jl")
    include("contractions/virtual.jl")
    #
    include("contractions_cuda/dense.jl")
    include("contractions_cuda/site.jl")
    include("contractions_cuda/sparse.jl")
    include("contractions_cuda/central.jl")
    include("contractions_cuda/diagonal.jl")
    include("contractions_cuda/virtual.jl")

end # module
