# contractions of dense objects on CPU and CUDA

const MatrixOrCuMatrix{R} = Union{CuArray{R, 2}, Diagonal{R, CuArray{R, 1, Mem.DeviceBuffer}}, Array{R, 2}, Diagonal{R, Vector{R}}}
const ArrayOrCuArray{R, N} = Union{Array{R, N}, CuArray{R, N}}

function contract_tensor3_matrix(A::ArrayOrCuArray{R, 3}, M::MatrixOrCuMatrix{R}) where R <: Real
    sl1, sl2, sl3 = size(A)
    A = reshape(A, sl1 * sl2, sl3)
    reshape(A * M, sl1, sl2, :)
end

function contract_matrix_tensor3(M::MatrixOrCuMatrix{R}, A::ArrayOrCuArray{R, 3}) where R <: Real
    sl1, sl2, sl3 = size(A)
    A = reshape(A, sl1 * sl2, sl3)
    reshape(A * M', sl1, sl2, :)
end

"""
        -- A --
      |    |
 L = LE -- M --
      |    |
        -- B --
"""
function update_env_left(LE::S, A::S, M::T, B::S) where {S <: ArrayOrCuArray{R, 3}, T <: ArrayOrCuArray{R, 4}} where R <: Real
    @tensor LE[nb, nt, nc] := LE[ob, ot, oc] * A[ot, nt, α] * M[oc, α, nc, β] * B[ob, nb, β] order = (ot, α, oc, β, ob)
end

"""
      -- A --
         |    |
 R =  -- M -- RE
         |    |
      -- B --
"""
function update_env_right(RE::S, A::S, M::T, B::S) where {T <: ArrayOrCuArray{R, 4}, S <: ArrayOrCuArray{R, 3}} where R <: Real
    @tensor RE[nb, nt, nc] := RE[ob, ot, oc] * A[nt, ot, α] * M[nc, α, oc, β] * B[nb, ob, β] order = (ot, α, oc, β, ob)
end

"""
   |    |    |
  LE -- M -- RE
   |    |    |
     -- B --
"""
function project_ket_on_bra(LE::S, B::S, M::T, RE::S) where {T <: ArrayOrCuArray{R, 4}, S <: ArrayOrCuArray{R, 3}} where R <: Real
    @tensor A[nl, nr, nc] := LE[ol, nl, lc] * B[ol, or, oc] * M[lc, nc, rc, oc] * RE[or, nr, rc] order = (ol, lc, oc, or, rc)
end

"""
      K
      |
   -- M -- RE
      |    |
   -- B ---
"""
function update_reduced_env_right(RE::ArrayOrCuArray{R, 2}, m::Int, M::MpoTensor{R, 4}, B::ArrayOrCuArray{R, 3}) where R <: Real
    K = zeros(R, size(M, 2))
    K[m] = one(R)
    K = reshape(CuArray(K), 1, 1, size(K, 1))
    for v ∈ M.top K = contract_tensor3_matrix(K, v) end
    K = dropdims(K, dims=(1, 2))

    for v ∈ reverse(M.bot)
        B = contract_matrix_tensor3(v, B)   # TODO do we ever enter here? in mpo layers that we have now, we don't
    end
    update_reduced_env_right(K, RE, M.ctr, B)
end

function update_reduced_env_right(K::CuArray{R, 1}, RE::ArrayOrCuArray{R, 2}, M::ArrayOrCuArray{R, 4}, B::ArrayOrCuArray{R, 3}) where R <: Real
    @tensor RE[x, y] := K[d] * M[y, d, β, γ] * B[x, α, γ] * RE[α, β] order = (d, β, γ, α)
end

function update_reduced_env_right(RR::S, M0::S) where S <: CuArray{<:Real, 2}
    @tensor RR[x, y] := M0[y, z] * RR[x, z]
end

function contract_tensors43(B::ArrayOrCuArray{R, 4}, A::ArrayOrCuArray{R, 3}) where R <: Real
    @matmul C[(x, y), (b, a), z] := sum(σ) B[y, z, a, σ] * A[x, b, σ]
end

function corner_matrix(C::S, M::T, B::S) where {S <: ArrayOrCuArray{R, 3}, T <: ArrayOrCuArray{R, 4}} where R <: Real
    @tensor Cnew[ll, ml, tt, mt] := M[ml, mt, mr, mb] * B[ll, rr, mb] * C[rr, tt, mr] order = (rr, mb, mr)
end
