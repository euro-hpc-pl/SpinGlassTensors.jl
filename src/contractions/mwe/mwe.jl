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

function prepare_projectors(::Type{T}) where T
    PoolOfProjectors{T}(
        Dict(:CPU => Dict(2 => [1, 1], 3 => [1], 1 => [1, 2]),
             :GPU => Dict(2 => [1, 1], 3 => [1], 1 => [1, 2]),
        ),
        :CPU,
        Dict(2 => 1, 3 => 1, 1 => 2),
    )
end

function prepare_virtual_tensor(::Type{T}; onGPU::Bool) where T
    simple_array = T[1.0;;]
    con = onGPU ? CuArray(simple_array) : simple_array
    M = VirtualTensor(
        prepare_projectors(Int),
        con,
        (1, 2, 2, 3, 3, 3),
        (2, 1, 1, 2),
    )
end

function prepare_input(::Type{T}; onGPU::Bool) where T
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

# This is problematic function (CPU - OK vs GPU - NOT OK)
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

    if srpb * slpt >= srpt * slpb # This loop is OK
        println("IF")
        B2 = permutedims(B, (1, 3, 2, 4))  # [lb, lpb, rb, rpb]
        B2 = reshape(B2, (slb * slpb, srb * srpb))  # [(lb, lpb), (rb, rpb)]

        pr_b_ct, pr_c_t, srpct =
            merge_projectors_inter(M.lp, p_rb, p_rc, p_rt, onGPU; order = "1_23")
        pl_bc_t, pl_b_c, slpbc =
            merge_projectors_inter(M.lp, p_lt, p_lb, p_lc, onGPU; order = "23_1")

        tmp1 = alloc_zeros(R, onGPU, (srb, srpb * srpct))
        tmp2 = alloc_undef(R, onGPU, (slb * slpb, srpct))
        tmp3 = alloc_zeros(R, onGPU, (slb * slpb, srpc * srpt))
        tmp5 = alloc_undef(R, onGPU, (slb * slpb, slpc, srpt))
        tmp8 = alloc_undef(R, onGPU, (slb * slpbc, slpt))

        for irt ∈ 1:srt
            tmp1[:, pr_b_ct] = (@view RE[:, irt, :])  # [rb, (rpb, rpct)]
            mul!(tmp2, B2, reshape(tmp1, (srb * srpb, srpct)))  # [(lb, lpb), rpct]
            tmp3[:, pr_c_t] = tmp2  # [(lb, lpb), (rpc, rpt)]
            tmp4 = reshape(tmp3, (slb * slpb, srpc, srpt))  # [(lb, lpb), rpc, rpt]
            batched_mul!(tmp5, tmp4, M.con')
            tmp6 = reshape(tmp5, (slb, slpb * slpc, srpt))  # [lb, (lpb, lpc), rpt]
            tmp7 = reshape(tmp6[:, pl_b_c, :], (slb * slpbc, srpt))  # [(lb, lpbc), rpt]
            for ilt ∈ 1:slt
                mul!(tmp8, tmp7, (@view A[ilt, irt, :, :])')
                tmp9 = reshape(tmp8, (slb, slpbc * slpt))
                Rout[:, ilt, :] .+= tmp9[:, pl_bc_t]
            end
        end
    else
        println("ELSE")
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
            tmp1[:, pr_t_cb] = (@view RE[irb, :, :])  # [rt, (rpt, rpcb)]
            mul!(tmp2, A2, reshape(tmp1, (srt * srpt, srpcb)))  # [(lt, lpt), rpcb]
            tmp3[:, pr_c_b] = tmp2  # [(lt, lpt), (rpc, rpb)]
            tmp4 = reshape(tmp3, (slt * slpt, srpc, srpb))  # [(lt, lpt), rpc, rpb]
            batched_mul!(tmp5, tmp4, M.con')  # [(lt, lpt), lpc, rpb]
            tmp6 = reshape(tmp5, (slt, slpt * slpc, srpb))  # [lt, (lpt, lpc), rpb]
            tmp7 = reshape(tmp6[:, pl_t_c, :], (slt * slptc, srpb))  # [(lb, lptc), rpb]
            for ilb ∈ 1:slb
                mul!(tmp8, tmp7, (@view B[ilb, irb, :, :])')
                tmp9 = reshape(tmp8, (slt, slptc * slpb))
                Rout[ilb, :, :] .+= tmp9[:, pl_tc_b]
            end
        end
    end
    Rout
end

# ============= MAIN =============

T = Float64

RE, A, M, B = prepare_input(T, onGPU = false)
Rout = problematic_update_env_right(RE, A, M, B)
println(Rout)
