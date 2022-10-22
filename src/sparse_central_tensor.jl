export 
    _update_tensor_forward_n,
    _update_tensor_forward_c,
    _update_tensor_backwards_n,
    _update_tensor_backwards_c
    
"""
$(TYPEDSIGNATURES)
"""
function update_env_left(
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

function _update_tensor_forward_n(
    C::T, B::S
    ) where {S <: AbstractArray{Float64, 3}, T <: SparseCentralTensor}
    MM = cuda_dense_central_tensor(C)
    B = CUDA.CuArray(B)
    @tensor B[l, x, r] := B[l, y, r] * MM[y, x]
    Array(B)
end

function _update_tensor_forward_c(
    C::T, B::S
    ) where {S <: AbstractArray{Float64, 3}, T <: SparseCentralTensor}
    MM = cuda_dense_central_tensor(C)
    B = CUDA.CuArray(B)
    @tensor B[l, x, r] := MM[x, y] * B[l, y, r]
    Array(B)
end

function _update_tensor_backwards_n(
    C::T, B::S
    ) where {S <: AbstractArray{Float64, 3}, T <: SparseCentralTensor}
    MM = cuda_dense_central_tensor(C)
    B = CUDA.CuArray(B)
    @tensor B[l, x, r] := B[l, y, r] * MM[x, y]
    Array(B)
end

function _update_tensor_backwards_c(
    C::T, B::S
    ) where {S <: AbstractArray{Float64, 3}, T <: SparseCentralTensor}
    MM = cuda_dense_central_tensor(C)
    B = CUDA.CuArray(B)
    @tensor B[l, x, r] := B[l, y, r] * MM[y, x]
    Array(B)
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_right(
    RE::S, M::T, ::Union{Val{:n}, Val{:c}}
) where {S <: AbstractArray{Float64, 3}, T <: Union{SparseCentralTensor, SparseVirtualTensor}}
    if typeof(M) == SparseCentralTensor
        MM = cuda_dense_central_tensor(M)
    else
        MM = CUDA.CuArray(M)
    end 
    RE = CUDA.CuArray(RE)
    @tensor R[nt, nc, nb] := MM[nc, oc] * RE[nt, oc, nb]
    Array(R)
end

"""
$(TYPEDSIGNATURES)
"""
function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{:n}
) where {T <: SparseCentralTensor, S <: AbstractArray{Float64, 3}}
    MM = cuda_dense_central_tensor(M)
    B = CUDA.CuArray(B)
    @tensor A[x, y, z] := MM[y, a] * B[x, a, z]
    Array(A)
end

"""
$(TYPEDSIGNATURES)
"""
function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{:c}
) where {T <: SparseCentralTensor, S <: AbstractArray{Float64, 3}}
    MM = cuda_dense_central_tensor(M)
    B = CUDA.CuArray(B)
    @tensor A[x, y, z] := B[x, a, z] * MM[a, y]
    Array(A)
end