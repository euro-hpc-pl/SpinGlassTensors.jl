function contract_tensor3_matrix(LE::CuArrayOrArray{T, 3}, M::CuArrayOrArray{T, 2}) where T <: Real
    if typeof(LE) <: CuArray && !(typeof(M) <: CuArray) M = CuArray(M) end
    @tensor L[nt, nc, nb] := LE[nt, oc, nb] * M[oc, nc]
end

function contract_matrix_tensor3(M::CuArrayOrArray{T, 2}, LE::CuArrayOrArray{T, 3}) where T <: Real
    if typeof(LE) <: CuArray && !(typeof(M) <: CuArray) M = CuArray(M) end
    @tensor L[nt, nc, nb] :=  LE[nt, oc, nb] * M[nc, oc]
end

"""
        -- A --
      |    |
 L = LE -- M --
      |    |
        -- B --
"""
function update_env_left(
    LE::S, A::S, M::T, B::S
) where {S <: CuArrayOrArray{R, 3}, T <: CuArrayOrArray{R, 4}} where R <: Real
    @tensor L[nb, nc, nt] := LE[ob, oc, ot] * A[ot, α, nt] *
                             M[oc, α, nc, β] * B[ob, β, nb] order = (ot, α, oc, β, ob)
end


"""
      -- A --
         |    |
 R =  -- M -- RE
         |    |
      -- B --
"""
function update_env_right(
    RE::S, A::S, M::T, B::S
) where {T <: CuArrayOrArray{F, 4}, S <: CuArrayOrArray{F, 3}} where F <: Real
    @tensor R[nt, nc, nb] := RE[ot, oc, ob] * A[nt, α, ot] *
                             M[nc, α, oc, β] * B[nb, β, ob] order = (ot, α, oc, β, ob)
end


"""
   |    |    |
  LE -- M -- RE
   |    |    |
     -- B --
"""
function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S
) where {T <: CuArrayOrArray{R, 4}, S <: CuArrayOrArray{R, 3}} where R <: Real
    @tensor A[x, y, z] := LE[k, l, x] * B[k, m, o] *
                          M[l, y, n, m] * RE[z, n, o] order = (k, l, m, n, o)
end

# function project_ket_on_bra(
#     LE::S, B::S, C::S, M::T, N::T, RE::S
# ) where {T <: CuArrayOrArray{R, 4}, S <: CuArrayOrArray{R, 3}} where R <: Real
#     @tensor A[x, y, z, r] := LE[k, l, x] * B[k, m, o] *
#                              M[l, y, n, m] * C[o, s, q] *
#                              N[n, z, p, s] * RE[r, p, q] order = (k, l, m, n, o, s, p, q)
# end

"""
      K
      |
   -- M -- RE
      |    |
   -- B ---
"""
function update_reduced_env_right(RE::Array{T, 2}, m::Int, M::MpoTensor{T, 4}, B::Array{T, 3}) where T <: Real
    K = zeros(T, size(M, 2))
    K[m] = one(T)
    K = reshape(K, 1, size(K, 1), 1)
    for v ∈ M.top
        K = contract_tensor3_matrix(K, v)
    end
    K = dropdims(K, dims=(1, 3))

    for v ∈ reverse(M.bot)
        B = contract_matrix_tensor3(v, B)   # do we ever enter here? in layers that we have now, likely not.
        println("do we ever enter here?")
    end
    update_reduced_env_right(K, RE, M.ctr, B)
end

function update_reduced_env_right(K::Array{T, 1}, RE::Array{T, 2}, M::Array{T, 4}, B::Array{T, 3}) where T <: Real
    @tensor R[x, y] := K[d] * M[y, d, β, γ] * B[x, γ, α] * RE[α, β] order = (d, β, γ, α)
end

function update_reduced_env_right(RR::S, M0::S) where S <: Array{<:Real, 2}
    @tensor RR[x, y] := M0[y, z] * RR[x, z]
end

function contract_tensors43(B::Array{T, 4}, A::Array{T, 3}) where T <: Real
    @matmul C[(x, y), z, (b, a)] := sum(σ) B[y, z, a, σ] * A[x, σ, b]
end
