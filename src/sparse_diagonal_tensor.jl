export 
    _update_tensor_forward_n,
    _update_tensor_forward_c,
    _update_tensor_backwards_n,
    _update_tensor_backwards_c

function _update_tensor_forward_n(
    C::T, B::S
    ) where {S <: AbstractArray{Float64, 3}, T <: SparseDiagonalTensor}
    @cast BB[l, s1, s2, r] := B[l, (s1, s2), r]  (s1 ∈ 1:size(C.e1, 1))
    @tensor CC[l, q2, q1, r] := BB[l, s1, s2, r] * C.e1[s1, q1] * C.e2[s2, q2]
    @cast CC[l, (q2, q1), r] := CC[l, q2, q1, r]
    CC
end

function _update_tensor_forward_c(
    C::T, B::S
    ) where {S <: AbstractArray{Float64, 3}, T <: SparseDiagonalTensor}
    @cast BB[l, s2, s1, r] := B[l, (s2, s1), r]  (s2 ∈ 1:size(C.e2, 2))
    @tensor CC[l, q1, q2, r] := C.e1[q1, s1] * C.e2[q2, s2] * BB[l, s2, s1, r]
    @cast CC[l, (q1, q2), r] := CC[l, q1, q2, r]
    CC
end

function _update_tensor_backwards_n(
    C::T, B::S
    ) where {S <: AbstractArray{Float64, 3}, T <: SparseDiagonalTensor}
    @cast BB[l, s2, s1, r] := B[l, (s2, s1), r]  (s2 ∈ 1:size(C.e2, 2))
    @tensor CC[l, q1, q2, r] := C.e1[q1, s1] * C.e2[q2, s2] * BB[l, s2, s1, r]
    @cast CC[l, (q1, q2), r] := CC[l, q1, q2, r]
    CC
end

function _update_tensor_backwards_c(
    C::T, B::S
    ) where {S <: AbstractArray{Float64, 3}, T <: SparseDiagonalTensor}
    @cast BB[l, s1, s2, r] := B[l, (s1, s2), r]  (s1 ∈ 1:size(C.e1, 1))
    @tensor CC[l, q2, q1, r] := BB[l, s1, s2, r] * C.e1[s1, q1] * C.e2[s2, q2]
    @cast CC[l, (q2, q1), r] := CC[l, q2, q1, r]
    CC
end

"""
$(TYPEDSIGNATURES)
"""
function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{:n} #TODO
) where {T <: SparseDiagonalTensor, S <: AbstractArray{Float64, 3}}
    @cast BB[l, s2, s1, r] := B[l, (s2, s1), r]  (s2 ∈ 1:size(M.e2, 2))
    @tensor CC[l, q1, q2, r] := M.e1[q1, s1] * M.e2[q2, s2] * BB[l, s2, s1, r]
    @cast CC[l, (q1, q2), r] := CC[l, q1, q2, r]
    CC
end

"""
$(TYPEDSIGNATURES)
"""
function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{:c}
) where {T <: SparseDiagonalTensor, S <: AbstractArray{Float64, 3}}
    @cast BB[l, s1, s2, r] := B[l, (s1, s2), r]  (s1 ∈ 1:size(M.e1, 1))
    @tensor CC[l, q2, q1, r] := BB[l, s1, s2, r] * M.e1[s1, q1] * M.e2[s2, q2]
    @cast CC[l, (q2, q1), r] := CC[l, q2, q1, r]
    CC
end