export 
    attach_central_left,
    attach_central_right
    
"""
$(TYPEDSIGNATURES)
"""
# function attach_central_left(
#     LE::S, M::T, ::Union{Val{:n}, Val{:c}}
# ) where {S <: AbstractArray{Float64, 3}, T <: SparseCentralTensor}
#     if typeof(LE) <: CuArray
#         MM = cuda_dense_central_tensor(M)
#     else
#         MM = dense_central_tensor(M)
#     end
#     @tensor L[nt, nc, nb] :=  LE[nt, oc, nb] * MM[oc, nc]
#     L
# end


function attach_central_left(
    LE::S, M::T, ::Union{Val{:n}, Val{:c}}
) where {S <: AbstractArray{Float64, 3}, T <: SparseCentralTensor}
    if typeof(LE) <: CuArray
        e11 = CUDA.CuArray(M.e11)
        e12 = CUDA.CuArray(M.e12)
        e21 = CUDA.CuArray(M.e21)
        e22 = CUDA.CuArray(M.e22)
    else
        e11, e12, e21, e22 = M.e11, M.e12, M.e21, M.e22
    end
    st, _, sb = size(LE)
    sl1 = size(e11, 1)
    sl2 = size(e21, 1)
    sr1 = size(e21, 2)
    sr2 = size(e22, 2)
    LE = permutedims(LE, (1, 3, 2))
    LE = reshape(LE, (st * sb, sl1, sl2))
    if sr1 <= sr2 && sl1 <= sl2
        #println("sr1 <= sr2 && sl1 <= sl2")
        @cast LE[tb, l1, l2, r1] := LE[tb, l1, l2] * e21[l2, r1]
        @tensor LE[tb, l1, r1, r2] := LE[tb, l1, l2, r1] * e22[l2, r2]
        @cast LE[tb, l1, r1, r2] := LE[tb, l1, r1, r2] * e11[l1, r1] * e12[l1, r2]
        LE = dropdims(sum(LE, dims=2), dims=2)
    elseif sr1 <= sr2 && sl1 > sl2
        #println("sr1 <= sr2 && sl1 > sl2")
        @cast LE[tb, l1, l2, r1] := LE[tb, l1, l2] * e11[l1, r1]
        @tensor LE[tb, l2, r1, r2] := LE[tb, l1, l2, r1] * e12[l1, r2]
        @cast LE[tb, l2, r1, r2] := LE[tb, l2, r1, r2] * e21[l2, r1] * e22[l2, r2]
        LE = dropdims(sum(LE, dims=2), dims=2)
    elseif sr1 > sr2 && sl1 <= sl2
        #println("sr1 > sr2 && sl1 <= sl2")
        @cast LE[tb, l1, l2, r2] := LE[tb, l1, l2] * e22[l2, r2]
        @tensor LE[tb, l1, r1, r2] := LE[tb, l1, l2, r2] * e21[l2, r1]
        @cast LE[tb, l1, r1, r2] := LE[tb, l1, r1, r2] * e11[l1, r1] * e12[l1, r2]
        LE = dropdims(sum(LE, dims=2), dims=2)
    else # sr1 > sr2 && sl1 > sl2
        #println("sr1 > sr2 && sl1 > sl2")
        @cast LE[tb, l1, l2, r2] := LE[tb, l1, l2] * e12[l1, r2]
        @tensor LE[tb, l2, r1, r2] := LE[tb, l1, l2, r2] * e11[l1, r1]
        @cast LE[tb, l2, r1, r2] := LE[tb, l2, r1, r2] * e21[l2, r1] * e22[l2, r2]
        LE = dropdims(sum(LE, dims=2), dims=2)
    end
    LE = reshape(LE, (st , sb, sr1 * sr2))
    LE = permutedims(LE, (1, 3, 2))
    LE
end


"""
$(TYPEDSIGNATURES)
"""
function attach_central_right(
    LE::S, M::T, ::Union{Val{:n}, Val{:c}}
) where {S <: AbstractArray{Float64, 3}, T <: SparseCentralTensor}
    if typeof(LE) <: CuArray
        MM = cuda_dense_central_tensor(M)
    else
        MM = dense_central_tensor(M)
    end
    @tensor L[nt, nc, nb] :=  LE[nt, oc, nb] * MM[nc, oc]
    L
end
