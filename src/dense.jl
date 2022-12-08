#TODO: clean Val{:n} and Val{:c}
function attach_central_left(LE::ArrayOrCuArray{3}, M::ArrayOrCuArray{2})
    if typeof(LE) <: CuArray && !(typeof(M) <: CuArray) M = CuArray(M) end
    @tensor L[nt, nc, nb] := LE[nt, oc, nb] * M[oc, nc]
    L
end

function attach_central_right(LE::ArrayOrCuArray{3}, M::ArrayOrCuArray{2})
    if typeof(LE) <: CuArray && !(typeof(M) <: CuArray) M = CuArray(M) end
    @tensor L[nt, nc, nb] :=  LE[nt, oc, nb] * M[nc, oc]
    L
end

function update_env_left(
    LE::S, A::S, M::T, B::S, ::Val{:n}
) where {S <: ArrayOrCuArray{3}, T <: ArrayOrCuArray{4}}
    @tensor L[nb, nc, nt] := LE[ob, oc, ot] * A[ot, α, nt] *
                             M[oc, α, nc, β] * B[ob, β, nb] order = (ot, α, oc, β, ob)
    L
end

function update_env_left(
    LE::S, A::S, M::T, B::S, ::Val{:c}
) where {S <: ArrayOrCuArray{3}, T <: ArrayOrCuArray{4}}
    @tensor L[nb, nc, nt] := LE[ob, oc, ot] * A[ot, α, nt] *
                             M[oc, β, nc, α] * B[ob, β, nb] order = (ot, α, oc, β, ob)
    L
end

"""
      -- A --
         |    |
 R =  -- M -- RE
         |    |
      -- B --
"""
function update_env_right(
    RE::S, A::S, M::T, B::S, ::Val{:n}
) where {T <: ArrayOrCuArray{4}, S <: ArrayOrCuArray{3}}
    @tensor R[nt, nc, nb] := RE[ot, oc, ob] * A[nt, α, ot] *
                             M[nc, α, oc, β] * B[nb, β, ob] order = (ot, α, oc, β, ob)
    R
end

function update_env_right(
    RE::S, A::S, M::T, B::S, ::Val{:c}
) where {T <: ArrayOrCuArray{4}, S <: ArrayOrCuArray{3}}
    @tensor R[nt, nc, nb] := RE[ot, oc, ob] * A[nt, α, ot] *
                             M[nc, β, oc, α] * B[nb, β, ob] order = (ot, α, oc, β, ob)
    R
end

"""
   |    |    |
  LE -- M -- RE
   |    |    |
     -- B --
"""
function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{:n}
) where {T <: ArrayOrCuArray{4}, S <: ArrayOrCuArray{3}}
    @tensor A[x, y, z] := LE[k, l, x] * B[k, m, o] *
                          M[l, y, n, m] * RE[z, n, o] order = (k, l, m, n, o)
    A
end

function project_ket_on_bra(
    LE::S, B::S, C::S, M::T, N::T, RE::S, ::Val{:n}
) where {T <: ArrayOrCuArray{4}, S <: ArrayOrCuArray{3}}
    @tensor A[x, y, z, r] := LE[k, l, x] * B[k, m, o] *
                          M[l, y, n, m] * C[o, s, q] *
                          N[n, z, p, s] * RE[r, p, q] order = (k, l, m, n, o, s, p, q)
    A
end

function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{:c}
) where {T <: ArrayOrCuArray{4}, S <: ArrayOrCuArray{3}}
    @tensor A[x, y, z] := LE[k, l, x] * B[k, m, o] *
                          M[l, m, n, y] * RE[z, n, o] order = (k, l, m, n, o)
    A
end

function project_ket_on_bra(
    LE::S, B::S, C::S, M::T, N::T, RE::S, ::Val{:c}
) where {T <: ArrayOrCuArray{4}, S <: ArrayOrCuArray{3}}
    @tensor A[x, m, s, r] := LE[k, l, x] * B[k, y, o] *
                          M[l, y, n, m] * C[o, z, q] *
                          N[n, z, p, s] * RE[r, p, q] order = (k, l, y, n, o, z, p, q)
    A
end


"""
      K
      |
   -- M -- RE
      |    |
   -- B ---
"""
function update_reduced_env_right(
    RE::AbstractArray{Float64, 2}, m::Int, M::Dict, B::AbstractArray{Float64, 3}
)
    kk = sort(collect(keys(M)))

    if kk[1] < 0
        K = zeros(size(M[kk[1]], 1))
        K[m] = 1.

        for ii ∈ kk[1:end]
            if ii == 0 break end
            Mm = M[ii]
            K = _project_on_border(K, Mm)
        end
    else
        K = zeros(size(M[0], 2))
        K[m] = 1.
    end

    update_reduced_env_right(K, RE, M[0], B)
end

function update_reduced_env_right(K::Array{T, 1}, RE::Array{T, 2}, M::Array{T, 4}, B::Array{T, 3}) where T <: Real
    @tensor R[x, y] := K[d] * M[y, d, β, γ] * B[x, γ, α] * RE[α, β] order = (d, β, γ, α)
    R
end

function update_reduced_env_right(RR::S, M0::S) where S <: AbstractArray{Float64, 2}
    @tensor RR[x, y] := M0[y, z] * RR[x, z]
    RR
end

function _project_on_border(K::S, M::T) where {S <: AbstractArray{Float64, 1}, T <: AbstractArray{Float64, 2}}
    @tensor K[a] := K[b] * M[b, a]
    K
end
