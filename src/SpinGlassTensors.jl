module SpinGlassTensors
    using LinearAlgebra
    using TensorOperations, TensorCast
    #using LowRankApprox
    using Memoize

    using DocStringExtensions

    include("base.jl")
    include("s_base.jl")
    include("linear_algebra_ext.jl")
    include("compressions.jl")
    #include("s_compressions.jl")
    include("identities.jl")
    #include("contractions.jl")
    include("s_contractions.jl")
end # module
