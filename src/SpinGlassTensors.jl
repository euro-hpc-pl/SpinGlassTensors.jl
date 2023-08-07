module SpinGlassTensors
    using LinearAlgebra, MKL
    using TensorOperations, TensorCast
    using LowRankApprox, TSVD
    using CUDA, CUDA.CUSPARSE
    using NNlib, NNlibCUDA
    using Memoization
    using SparseArrays
    using SpinGlassNetworks

    CUDA.allowscalar(false)

    include("projectors.jl")
    include("base.jl")
    include("linear_algebra_ext.jl")
    include("./mps/base.jl")
    include("./mps/transpose.jl")
    include("./mps/dot.jl")
    include("./mps/identity.jl")
    include("./mps/aux.jl")
    include("./mps/rand.jl")
    include("transfer.jl")
    include("environment.jl")
    include("utils/memory.jl")
    include("./mps/canonise.jl")
    include("variational.jl")
    include("zipper.jl")
    include("gauges.jl")
    include("contractions/sparse.jl")
    include("contractions/dense.jl")
    include("contractions/central.jl")
    include("contractions/diagonal.jl")
    include("contractions/site.jl")
    include("contractions/virtual.jl")

end # module
