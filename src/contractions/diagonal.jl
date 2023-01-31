# contractions with DiagonalTensor on CPU and CUDA

function contract_tensor3_matrix(B::Union{Array{R, 3}, CuArray{R, 3}}, C::DiagonalTensor{R}) where R <: Real
    @cast B[l, (r, s1), s2] := B[l, r, (s1, s2)] (s2 ∈ 1:size(C.e2, 1))
    B = contract_tensor3_matrix(B, C.e2)
    @cast B[l, r, s1, q2] := B[l, (r, s1), q2] (s1 ∈ 1:size(C.e1, 1))
    B = permutedims(B, (1, 2, 4, 3))
    @cast B[l, (r, q2), s1] := B[l, r, q2, s1]
    B = contract_tensor3_matrix(B, C.e1)
    @cast B[l, r, (q2, q1)] := B[l, (r, q2), q1] (q2 ∈ 1:size(C.e2, 2))
end

function contract_matrix_tensor3(C::DiagonalTensor{R}, B::Union{Array{R, 3}, CuArray{R, 3}}) where R <: Real
    @cast B[l, (r, s2), s1] := B[l, r, (s2, s1)] (s1 ∈ 1:size(C.e1, 2))
    B = contract_matrix_tensor3(C.e1, B)
    @cast B[l, r, s2, q1] := B[l, (r, s2), q1] (s2 ∈ 1:size(C.e2, 2))
    B = permutedims(B, (1, 2, 4, 3))
    @cast B[l, (r, q1), s2] := B[l, r, q1, s2]
    B = contract_matrix_tensor3(C.e2, B)
    @cast B[l, r, (q1, q2)] := B[l, (r, q1), q2] (q1 ∈ 1:size(C.e1, 1))
end
