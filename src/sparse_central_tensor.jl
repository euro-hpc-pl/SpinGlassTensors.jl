export 
    attach_central_left,
    attach_central_right
    
"""
$(TYPEDSIGNATURES)
"""
function attach_central_left(
    LE::S, M::T, ::Union{Val{:n}, Val{:c}}
) where {S <: AbstractArray{Float64, 3}, T <: SparseCentralTensor}
    if typeof(LE) <: CuArray
        MM = cuda_dense_central_tensor(M)
    else
        MM = dense_central_tensor(M)
    end
    @tensor L[nt, nc, nb] :=  LE[nt, oc, nb] * MM[oc, nc]
    L
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
