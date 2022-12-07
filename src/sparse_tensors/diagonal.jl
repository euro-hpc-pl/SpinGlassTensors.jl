function attach_central_left(B::Array{<:Real, 3}, C::SparseDiagonalTensor)
    @cast BB[l, s1, (s2, r)] := B[l, (s1, s2), r] (s2 ∈ 1:size(C.e2, 1))
    BB = attach_central_left(BB, C.e1)
    @cast BB[(l, q1), s2, r] := BB[l, q1, (s2, r)] (s2 ∈ 1:size(C.e2, 1))
    BB = attach_central_left(BB, C.e2)
    @cast BB[l, q1, q2, r] := BB[(l, q1), q2, r] (q1 ∈ 1:size(C.e1, 2))
    BB = permutedims(BB, (1, 3, 2, 4))
    @cast BB[l, (q2, q1), r] := BB[l, q2, q1, r]
    BB
end

function attach_central_right(B::Array{<:Real, 3}, C::SparseDiagonalTensor)
    @cast BB[l, s2, (s1, r)] := B[l, (s2, s1), r] (s1 ∈ 1:size(C.e1, 2))
    BB = attach_central_right(BB, C.e2)
    @cast BB[(l, q2), s1, r] := BB[l, q2, (s1, r)] (s1 ∈ 1:size(C.e1, 2))
    BB = attach_central_right(BB, C.e1)
    @cast BB[l, q2, q1, r] := BB[(l, q2), q1, r] (q2 ∈ 1:size(C.e2, 1))
    BB = permutedims(BB, (1, 3, 2, 4))
    @cast BB[l, (q1, q2), r] := BB[l, q1, q2, r]
    BB
end

function _project_on_border(
    K::S, M::T
    ) where {S <: AbstractArray{Float64, 1}, T <: SparseDiagonalTensor}
    sa, sb = size(M.e1, 1), size(M.e2, 1)
    qa, qb = size(M.e1, 2), size(M.e2, 2)
    K = reshape(K, 1, sa, sb)
    K = attach_central_left(K, M.e1)
    K = reshape(K, qa, sb, 1)
    K = attach_central_left(K, M.e2)
    K = reshape(K, qa, qb)
    K = permutedims(K, (2, 1))
    K = reshape(K, qa * qb)
    K
end