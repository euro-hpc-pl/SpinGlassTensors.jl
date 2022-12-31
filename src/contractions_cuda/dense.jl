const CuMatrix{T} = Union{CuArray{T, 2}, Diagonal{T, CuArray{T, 1, Mem.DeviceBuffer}}}

function contract_tensor3_matrix(LE::CuArray{T, 3}, M::CuMatrix{T}) where T <: Real
    sl1, sl2, sl3 = size(LE)
    LE = reshape(permutedims(LE, (1, 3, 2)), sl1 * sl3, sl2)
    permutedims(reshape(LE * M, sl1, sl3, :), (1, 3, 2))
end

function contract_matrix_tensor3(M::CuMatrix{T}, LE::CuArray{T, 3}) where T <: Real
    sl1, sl2, sl3 = size(LE)
    LE = reshape(permutedims(LE, (2, 1, 3)), sl2, sl1 * sl3)
    permutedims(reshape(M * LE, :, sl1, sl3), (2, 1, 3))
end

"""
        -- A --
      |    |
 L = LE -- M --
      |    |
        -- B --
"""
function update_env_left(LE::S, A::S, M::T, B::S) where {S <: CuArray{R, 3}, T <: CuArray{R, 4}} where R <: Real
    @tensor L[nb, nc, nt] := LE[ob, oc, ot] * A[ot, α, nt] * M[oc, α, nc, β] * B[ob, β, nb] order = (ot, α, oc, β, ob)
end

"""
      -- A --
         |    |
 R =  -- M -- RE
         |    |
      -- B --
"""
function update_env_right(RE::S, A::S, M::T, B::S) where {T <: CuArray{F, 4}, S <: CuArray{F, 3}} where F <: Real
    @tensor R[nt, nc, nb] := RE[ot, oc, ob] * A[nt, α, ot] * M[nc, α, oc, β] * B[nb, β, ob] order = (ot, α, oc, β, ob)
end

"""
   |    |    |
  LE -- M -- RE
   |    |    |
     -- B --
"""
function project_ket_on_bra(LE::S, B::S, M::T, RE::S) where {T <: CuArray{R, 4}, S <: CuArray{R, 3}} where R <: Real
    @tensor A[x, y, z] := LE[k, l, x] * B[k, m, o] * M[l, y, n, m] * RE[z, n, o] order = (k, l, m, n, o)
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
    K = reshape(CuArray(K), 1, size(K, 1), 1)
    for v ∈ M.top K = contract_tensor3_matrix(K, v) end
    K = dropdims(K, dims=(1, 3))

    for v ∈ reverse(M.bot)
        B = contract_matrix_tensor3(v, B)   # TODO do we ever enter here? in layers that we have now, likely not.
        println("do we ever enter here?")
    end
    update_reduced_env_right(K, RE, M.ctr, B)
end

function update_reduced_env_right(K::CuArray{T, 1}, RE::CuArray{T, 2}, M::CuArray{T, 4}, B::CuArray{T, 3}) where T <: Real
    @tensor R[x, y] := K[d] * M[y, d, β, γ] * B[x, γ, α] * RE[α, β] order = (d, β, γ, α)
end

function update_reduced_env_right(RR::S, M0::S) where S <: CuArray{<:Real, 2}
    @tensor RR[x, y] := M0[y, z] * RR[x, z]
end

function contract_tensors43(B::CuArray{T, 4}, A::CuArray{T, 3}) where T <: Real
    @matmul C[(x, y), z, (b, a)] := sum(σ) B[y, z, a, σ] * A[x, σ, b]
end

function corner_matrix(C::S, M::T, B::S) where {S <: CuArray{R, 3}, T <: CuArray{R, 4}} where R <: Real
    @tensor Cnew[l, ml, mt, tt] := M[ml, mt, mr, mb] * B[l, mb, r] * C[tt, mr, r] order = (r, mb, mr)
end
