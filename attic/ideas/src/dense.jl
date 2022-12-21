function attach_central_left(LE::ArrayOrCuArray{T, 3}, M::ArrayOrCuArray{T, 2}) where T <: Real
    typeof(LE) <: CuArray && !(typeof(M) <: CuArray) M = CuArray(M)
    @tensor L[nt, nc, nb] := LE[nt, oc, nb] * M[oc, nc]
end

function attach_central_right(LE::ArrayOrCuArray{T, 3}, M::ArrayOrCuArray{T, 2}) where T <: Real
    typeof(LE) <: CuArray && !(typeof(M) <: CuArray) M = CuArray(M)
    @tensor L[nt, nc, nb] :=  LE[nt, oc, nb] * M[nc, oc]
end

function update_env_left(
    LE::S, A::S, M::T, B::S, ::Val{false}
) where {S <: ArrayOrCuArray{R, 3}, T <: ArrayOrCuArray{R, 4} where R <: Real}
    @tensor L[nb, nc, nt] := LE[ob, oc, ot] * A[ot, α, nt] *
                             M[oc, α, nc, β] * B[ob, β, nb] order = (ot, α, oc, β, ob)
end

function update_env_left(
    LE::S, A::S, M::T, B::S, ::Val{true}
) where {S <: ArrayOrCuArray{R, 3}, T <: ArrayOrCuArray{R, 4} where R <: Real}
    @tensor L[nb, nc, nt] := LE[ob, oc, ot] * A[ot, α, nt] *
                             M[oc, β, nc, α] * B[ob, β, nb] order = (ot, α, oc, β, ob)
end

"""
      -- A --
         |    |
 R =  -- M -- RE
         |    |
      -- B --
"""
function update_env_right(
    RE::S, A::S, M::T, B::S, ::Val{false}
) where {T <: ArrayOrCuArray{R, 4}, S <: ArrayOrCuArray{R, 3} where R <: Real}
    @tensor R[nt, nc, nb] := RE[ot, oc, ob] * A[nt, α, ot] *
                             M[nc, α, oc, β] * B[nb, β, ob] order = (ot, α, oc, β, ob)
end

function update_env_right(
    RE::S, A::S, M::T, B::S, ::Val{true}
) where {T <: ArrayOrCuArray{R, 4}, S <: ArrayOrCuArray{R, 3} where R <: Real}
    @tensor R[nt, nc, nb] := RE[ot, oc, ob] * A[nt, α, ot] *
                             M[nc, β, oc, α] * B[nb, β, ob] order = (ot, α, oc, β, ob)
end

"""
   |    |    |
  LE -- M -- RE
   |    |    |
     -- B --
"""
function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{false} 
) where {T <: ArrayOrCuArray{R, 4}, S <: ArrayOrCuArray{R, 3} where R <: Real}
    @tensor A[x, y, z] := LE[k, l, x] * B[k, m, o] * M[l, y, n, m] * RE[z, n, o] order = (k, l, m, n, o)
end

function project_ket_on_bra(
    LE::S, B::S, C::S, M::T, N::T, RE::S, ::Val{false}
) where {T <: ArrayOrCuArray{R, 4}, S <: ArrayOrCuArray{R, 3} where R <: Real}
    @tensor A[x, y, z, r] := LE[k, l, x] * B[k, m, o] * M[l, y, n, m] * C[o, s, q] *
                             N[n, z, p, s] * RE[r, p, q] order = (k, l, m, n, o, s, p, q)
end

function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{true}
) where {T <: ArrayOrCuArray{R, 4}, S <: ArrayOrCuArray{R, 3} where R <: Real}
    @tensor A[x, y, z] := LE[k, l, x] * B[k, m, o] * M[l, m, n, y] * RE[z, n, o] order = (k, l, m, n, o)
end

function project_ket_on_bra(
    LE::S, B::S, C::S, M::T, N::T, RE::S, ::Val{true}
) where {T <: ArrayOrCuArray{R, 4}, S <: ArrayOrCuArray{R, 3} where R <: Real}
    @tensor A[x, m, s, r] := LE[k, l, x] * B[k, y, o] * M[l, y, n, m] * C[o, z, q] *
                             N[n, z, p, s] * RE[r, p, q] order = (k, l, y, n, o, z, p, q)
end

"""
      K
      |
   -- M -- RE
      |    |
   -- B ---
"""
function update_reduced_env_right(RE::Array{T, 2}, m::Int, M::Dict, B::Array{T, 3}) where T <: Real
    kk = sort(collect(keys(M)))
    if kk[1] < 0
        K = zeros(T, size(M[kk[1]], 1))
        K[m] = one(T)
        for ii ∈ kk[1:end]
            ii == 0 && break
            K = _project_on_border(K, M[ii])
        end
    else
        K = zeros(T, size(M[0], 2))
        K[m] = one(T)
    end
    update_reduced_env_right(K, RE, M[0], B)
end

function update_reduced_env_right(
    K::Array{T, 1}, RE::Array{T, 2}, M::Array{T, 4}, B::Array{T, 3}
) where T <: Real
    @tensor R[x, y] := K[d] * M[y, d, β, γ] * B[x, γ, α] * RE[α, β] order = (d, β, γ, α)
end

function update_reduced_env_right(RR::S, M0::S) where S <: Array{<:Real, 2}
    @tensor RR[x, y] := M0[y, z] * RR[x, z]
end

function _project_on_border(K::S, M::T) where {S <: Array{R, 1}, T <: Array{R, 2} where R <: Real}
    @tensor K[a] := K[b] * M[b, a]
end
