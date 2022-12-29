function contract_tensor3_matrix(B::CuArray{T, 3}, C::DiagonalTensor{T}) where T <: Real
    @cast BB[l, s1, (s2, r)] := B[l, (s1, s2), r] (s2 ∈ 1:size(C.e2, 1))
    BB = contract_tensor3_matrix(BB, C.e1)
    @cast BB[(l, q1), s2, r] := BB[l, q1, (s2, r)] (s2 ∈ 1:size(C.e2, 1))
    BB = contract_tensor3_matrix(BB, C.e2)
    @cast BB[l, q1, q2, r] := BB[(l, q1), q2, r] (q1 ∈ 1:size(C.e1, 2))
    BB = permutedims(BB, (1, 3, 2, 4))
    @cast BB[l, (q2, q1), r] := BB[l, q2, q1, r]
end

function contract_matrix_tensor3(C::DiagonalTensor{T}, B::CuArray{T, 3}) where T <: Real
    @cast BB[l, s2, (s1, r)] := B[l, (s2, s1), r] (s1 ∈ 1:size(C.e1, 2))
    BB = contract_matrix_tensor3(C.e2, BB)
    @cast BB[(l, q2), s1, r] := BB[l, q2, (s1, r)] (s1 ∈ 1:size(C.e1, 2))
    BB = contract_matrix_tensor3(C.e1, BB)
    @cast BB[l, q2, q1, r] := BB[(l, q2), q1, r] (q2 ∈ 1:size(C.e2, 1))
    BB = permutedims(BB, (1, 3, 2, 4))
    @cast BB[l, (q1, q2), r] := BB[l, q1, q2, r]
end
