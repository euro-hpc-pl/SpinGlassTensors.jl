export 
    attach_central_left,
    attach_central_right
    
"""
$(TYPEDSIGNATURES)
"""
function attach_central_left(
    LE::S, M::T, ::Union{Val{:n}, Val{:c}}
) where {S <: AbstractArray{Float64, 3}, T <: Union{SparseCentralTensor, SparseVirtualTensor}}
    if typeof(M) == SparseCentralTensor
        MM = cuda_dense_central_tensor(M)
    else
        MM = CUDA.CuArray(M)
    end    
    LE = CUDA.CuArray(LE)
    @tensor L[nt, nc, nb] :=  LE[nt, oc, nb] * MM[oc, nc]
    Array(L)
end

"""
$(TYPEDSIGNATURES)
"""
function attach_central_right(
    LE::S, M::T, ::Union{Val{:n}, Val{:c}}
) where {S <: AbstractArray{Float64, 3}, T <: Union{SparseCentralTensor, SparseVirtualTensor}}
    if typeof(M) == SparseCentralTensor
        MM = cuda_dense_central_tensor(M)
    else
        MM = CUDA.CuArray(M)
    end    
    LE = CUDA.CuArray(LE)
    @tensor L[nt, nc, nb] :=  LE[nt, oc, nb] * MM[nc, oc]
    Array(L)
end
