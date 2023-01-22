const CuMatrix{T} = Union{CuArray{T, 2}, Diagonal{T, CuArray{T, 1, Mem.DeviceBuffer}}}

function contract_tensor3_matrix(A::CuArray{T, 3}, M::CuMatrix{T}) where T <: Real
    sl1, sl2, sl3 = size(A)
    A = reshape(A, sl1 * sl2, sl3)
    reshape(A * M, sl1, sl2, :)
    # @tensor A[l, r, t] := A[l, r, ot] * M[ot, t]
end

function contract_matrix_tensor3(M::CuMatrix{T}, A::CuArray{T, 3}) where T <: Real
    sl1, sl2, sl3 = size(A)
    A = reshape(A, sl1 * sl2, sl3)
    reshape(A * M', sl1, sl2, :)
    # @tensor A[l, r, t] := M[t, ot] * A[l, r, ot]
end

"""
        -- A --
      |    |
 L = LE -- M --
      |    |
        -- B --
"""
function update_env_left(LE::S, A::S, M::T, B::S) where {S <: CuArray{R, 3}, T <: CuArray{R, 4}} where R <: Real
    @tensor L[nb, nt, nc] := LE[ob, ot, oc] * A[ot, nt, α] * M[oc, α, nc, β] * B[ob, nb, β] order = (ot, α, oc, β, ob)
end

"""
      -- A --
         |    |
 R =  -- M -- RE
         |    |
      -- B --
"""
function update_env_right(RE::S, A::S, M::T, B::S) where {T <: CuArray{F, 4}, S <: CuArray{F, 3}} where F <: Real
    @tensor R[nb, nt, nc] := RE[ob, ot, oc] * A[nt, ot, α] * M[nc, α, oc, β] * B[nb, ob, β] order = (ot, α, oc, β, ob)
end

"""
   |    |    |
  LE -- M -- RE
   |    |    |
     -- B --
"""
function project_ket_on_bra(LE::S, B::S, M::T, RE::S) where {T <: CuArray{R, 4}, S <: CuArray{R, 3}} where R <: Real
    @tensor A[nl, nr, nc] := LE[ol, nl, lc] * B[ol, or, oc] * M[lc, nc, rc, oc] * RE[or, nr, rc] order = (ol, lc, oc, or, rc)
end

"""
      K
      |
   -- M -- RE
      |    |
   -- B ---
"""
function update_reduced_env_right(RE::CuArray{T, 2}, m::Int, M::MpoTensor{T, 4}, B::CuArray{T, 3}) where T <: Real
    K = zeros(T, size(M, 2))
    K[m] = one(T)
    K = reshape(CuArray(K), 1, 1, size(K, 1))
    for v ∈ M.top K = contract_tensor3_matrix(K, v) end
    K = dropdims(K, dims=(1, 2))

    for v ∈ reverse(M.bot)
        B = contract_matrix_tensor3(v, B)   # TODO do we ever enter here? in layers that we have now, likely not.
        println("do we ever enter here?")
    end
    update_reduced_env_right(K, RE, M.ctr, B)
end

function update_reduced_env_right(K::CuArray{T, 1}, RE::CuArray{T, 2}, M::CuArray{T, 4}, B::CuArray{T, 3}) where T <: Real
    @tensor R[x, y] := K[d] * M[y, d, β, γ] * B[x, α, γ] * RE[α, β] order = (d, β, γ, α)
end

function update_reduced_env_right(RR::S, M0::S) where S <: CuArray{<:Real, 2}
    @tensor RR[x, y] := M0[y, z] * RR[x, z]
end

function contract_tensors43(B::CuArray{T, 4}, A::CuArray{T, 3}) where T <: Real
    @matmul C[(x, y), (b, a), z] := sum(σ) B[y, z, a, σ] * A[x, b, σ]
end

function corner_matrix(C::S, M::T, B::S) where {S <: CuArray{R, 3}, T <: CuArray{R, 4}} where R <: Real
    @tensor Cnew[ll, ml, tt, mt] := M[ml, mt, mr, mb] * B[ll, rr, mb] * C[rr, tt, mr] order = (rr, mb, mr)
end
