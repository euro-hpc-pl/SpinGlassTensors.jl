# function attach_central_left(
#     B::S, C::T, ::Union{Val{:n}, Val{:c}}
#     ) where {S <: Array{Float64, 3}, T <: SparseDiagonalTensor}
#     @cast BB[l, s1, s2, r] := B[l, (s1, s2), r]  (s1 ∈ 1:size(C.e1, 1))
#     @tensor CC[l, q2, q1, r] := BB[l, s1, s2, r] * C.e1[s1, q1] * C.e2[s2, q2]
#     @cast CC[l, (q2, q1), r] := CC[l, q2, q1, r]
#     CC
# end

# function attach_central_right(
#     B::S, C::T, ::Union{Val{:n}, Val{:c}}
#     ) where {S <: Array{Float64, 3}, T <: SparseDiagonalTensor}
#     @cast BB[l, s2, s1, r] := B[l, (s2, s1), r]  (s2 ∈ 1:size(C.e2, 2))
#     @tensor CC[l, q1, q2, r] := C.e1[q1, s1] * C.e2[q2, s2] * BB[l, s2, s1, r]
#     @cast CC[l, (q1, q2), r] := CC[l, q1, q2, r]
#     CC
# end


function attach_central_left(
    B::S, C::T
    ) where {S <: Array{Float64, 3}, T <: SparseDiagonalTensor}
    @cast BB[l, s1, (s2, r)] := B[l, (s1, s2), r]  (s2 ∈ 1:size(C.e2, 1))
    BB = attach_central_left(BB, C.e1)
    @cast BB[(l, q1), s2, r] := BB[l, q1, (s2, r)]  (s2 ∈ 1:size(C.e2, 1))
    BB = attach_central_left(BB, C.e2)
    @cast BB[l, q1, q2, r] := BB[(l, q1), q2, r]  (q1 ∈ 1:size(C.e1, 2))
    BB = permutedims(BB, (1, 3, 2, 4))
    @cast BB[l, (q2, q1), r] := BB[l, q2, q1, r]
    BB
end

function attach_central_right(
    B::S, C::T
    ) where {S <: Array{Float64, 3}, T <: SparseDiagonalTensor}
    @cast BB[l, s2, (s1, r)] := B[l, (s2, s1), r]  (s1 ∈ 1:size(C.e1, 2))
    BB = attach_central_right(BB, C.e2)
    @cast BB[(l, q2), s1, r] := BB[l, q2, (s1, r)]  (s1 ∈ 1:size(C.e1, 2))
    BB = attach_central_right(BB, C.e1)
    @cast BB[l, q2, q1, r] := BB[(l, q2), q1, r]  (q2 ∈ 1:size(C.e2, 1))
    BB = permutedims(BB, (1, 3, 2, 4))
    @cast BB[l, (q1, q2), r] := BB[l, q1, q2, r]
    BB
end
