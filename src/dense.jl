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
