using cuTENSOR
using CUDA, CUDA.CUSPARSE
using NNlib
using LinearAlgebra, MKL
using TensorOperations
using DocStringExtensions
using Base.Cartesian

include("../../projectors.jl")
include("../../base.jl")
include("../virtual.jl")
include("../central.jl")
include("../../utils/utils.jl")

ArrayorCuArray(A::AbstractArray, onGPU) = onGPU ? CuArray(A) : A

function VirtualTensor(lp, con, projs, dims)
    T = eltype(con)
    VirtualTensor{T, 4}(lp, con, projs, dims)
end

function prepare_projectors(::Type{T}; onGPU::Bool = true) where T
    if onGPU
        dict = Dict(
                :GPU => Dict(2 => CuArray([1, 1]), 3 => CuArray([1]), 1 => CuArray([1, 2])),
            )
    else
        dict = Dict(
            :CPU => Dict(2 => [1, 1], 3 => [1], 1 => [1, 2]),
            )
    end

    PoolOfProjectors{T}(
        dict,
        onGPU ? :GPU : :CPU,
        Dict(2 => 1, 3 => 1, 1 => 2),
    )
end

function prepare_virtual_tensor(::Type{T}; onGPU::Bool = true) where T
    simple_array = T[1.0;;]
    con = onGPU ? CuArray(simple_array) : simple_array
    M = VirtualTensor(
        prepare_projectors(Int; onGPU = onGPU),
        con,
        (1, 2, 2, 3, 3, 3),
        (2, 1, 1, 2),
    )
end

function prepare_input(::Type{T}; onGPU::Bool = true) where T
    # Data for which we found that the problem occurs
    RE = T[1.0; 0.9999999999999997;;;]
    A = T[1.0;;;]
    B = T[0.13466673434714854 0.9950796279655743; 0.013435384215150292 -0.09907842352346106;;;
         0.9950600547508714 0.13466938329370418; 0.09927480767643285 -0.013408806510184675
        ]

    M = prepare_virtual_tensor(T; onGPU = onGPU)

    if onGPU
        RE = CuArray(RE)
        A = CuArray(A)
        B = CuArray(B)
    end
    RE, A, M, B
end

# This is the problematic function (CPU - OK vs GPU - NOT OK)
function problematic_update_env_right(
    RE::S,
    A::S,
    M::VirtualTensor{R,4},
    B::S,
) where {S<:Tensor{R,3}} where {R<:Real}

    p_lb, p_lc, p_lt, p_rb, p_rc, p_rt = M.projs

    slb, srb = size(B, 1), size(B, 2)
    slt, srt = size(A, 1), size(A, 2)
    slc = length(M.lp, p_lc)

    slpb, slpc, slpt = size(M.lp, p_lb), size(M.lp, p_lc), size(M.lp, p_lt)
    srpb, srpc, srpt = size(M.lp, p_rb), size(M.lp, p_rc), size(M.lp, p_rt)

    onGPU = typeof(RE) <: CuArray

    A = reshape(A, (slt, srt, slpt, srpt))
    B = reshape(B, (slb, srb, slpb, srpb))
    Rout = alloc_zeros(R, onGPU, (slb, slt, slc))

    A2 = permutedims(A, (1, 3, 2, 4))  # [lt, lpt, rt, rpt]
    A2 = reshape(A2, (slt * slpt, srt * srpt))  # [(lt, lpt), (rt, rpt)]

    pr_t_cb, pr_c_b, srpcb =
        merge_projectors_inter(M.lp, p_rt, p_rc, p_rb, onGPU; order = "1_23")
    pl_tc_b, pl_t_c, slptc =
        merge_projectors_inter(M.lp, p_lb, p_lt, p_lc, onGPU; order = "23_1")

    tmp1 = alloc_zeros(R, onGPU, (srt, srpt * srpcb))
    tmp2 = alloc_undef(R, onGPU, (slt * slpt, srpcb))
    tmp3 = alloc_zeros(R, onGPU, (slt * slpt, srpc * srpb))
    tmp5 = alloc_undef(R, onGPU, (slt * slpt, slpc, srpb))
    tmp8 = alloc_undef(R, onGPU, (slt * slptc, slpb))

    for irb ∈ 1:srb
        tmp1[:, pr_t_cb] = (@view RE[irb, :, :])  # [rt, (rpt, rpcb)] # This line is (probably also) problematic
        mul!(tmp2, A2, reshape(tmp1, (srt * srpt, srpcb)))  # [(lt, lpt), rpcb]
        tmp3[:, pr_c_b] = tmp2  # [(lt, lpt), (rpc, rpb)]
        tmp4 = reshape(tmp3, (slt * slpt, srpc, srpb))  # [(lt, lpt), rpc, rpb]
        batched_mul!(tmp5, tmp4, M.con')  # [(lt, lpt), lpc, rpb]
        tmp6 = reshape(tmp5, (slt, slpt * slpc, srpb))  # [lt, (lpt, lpc), rpb]
        tmp7 = reshape(tmp6[:, pl_t_c, :], (slt * slptc, srpb))  # [(lb, lptc), rpb]

        for ilb ∈ 1:slb
            mul!(tmp8, tmp7, (@view B[ilb, irb, :, :])') # This line is problematic
            #mul!(tmp8, tmp7, B[ilb, irb, :, :]')
            tmp9 = reshape(tmp8, (slt, slptc * slpb))
            Rout[ilb, :, :] .+= tmp9[:, pl_tc_b]
        end
    end

    Rout
end

# ================ MAIN ================
# CPU and GPU should give the same result

T = Float64

RE, A, M, B = prepare_input(T, onGPU = false)
Rout = problematic_update_env_right(RE, A, M, B)

println("CPU R (OK):")
println(Rout)

RE, A, M, B = prepare_input(T, onGPU = true)
Rout = problematic_update_env_right(RE, A, M, B)

println("GPU R (NOT OK):")
println(Rout)
