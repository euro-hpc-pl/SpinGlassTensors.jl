module SpinGlassTensors
    using LinearAlgebra
    using TensorOperations, TensorCast
    using LowRankApprox
    using Memoize

    using DocStringExtensions

    include("base.jl")
    include("linear_algebra_ext.jl")
    include("compressions.jl")
    include("identities.jl")
    include("contractions.jl")
end # module
