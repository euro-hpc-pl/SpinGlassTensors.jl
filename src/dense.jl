export
    _update_tensor_forward_n,
    _update_tensor_forward_c,
    _update_tensor_backwards_n,
    _update_tensor_backwards_c

"""
$(TYPEDSIGNATURES)
"""
function update_env_left(
    LE::S, M::T, ::Union{Val{:n}, Val{:c}}
) where {S <: AbstractArray{Float64, 3}, T <: AbstractArray{Float64, 2}}
    @tensor L[nt, nc, nb] :=  LE[nt, oc, nb] * M[oc, nc]
    L
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_left(
    LE::S, A::S, M::T, B::S, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: AbstractArray{Float64, 4}}
    @tensor L[nb, nc, nt] := LE[ob, oc, ot] * A[ot, α, nt] *
                             M[oc, α, nc, β] * B[ob, β, nb] order = (ot, α, oc, β, ob)
    L
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_left(
    LE::S, A::S, M::T, B::S, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: AbstractArray{Float64, 4}}
    @tensor L[nb, nc, nt] := LE[ob, oc, ot] * A[ot, α, nt] *
                             M[oc, β, nc, α] * B[ob, β, nb] order = (ot, α, oc, β, ob)
    L
end

function _update_tensor_forward_n(
    C::T, B::S
    ) where {S <: AbstractArray{Float64, 3}, T <: AbstractArray{Float64, 2}}
    @tensor B[l, x, r] := B[l, y, r] * C[y, x]
    B
end

function _update_tensor_forward_c(
    C::T, B::S
    ) where {S <: AbstractArray{Float64, 3}, T <: AbstractArray{Float64, 2}}
    @tensor B[l, x, r] := B[l, y, r] * C[x, y]
    B
end

function _update_tensor_backwards_n(
    C::T, B::S
    ) where {S <: AbstractArray{Float64, 3}, T <: AbstractArray{Float64, 2}}
    @tensor B[l, x, r] := B[l, y, r] * C[x, y]
end

function _update_tensor_backwards_c(
    C::T, B::S
    ) where {S <: AbstractArray{Float64, 3}, T <: AbstractArray{Float64, 2}}
    @tensor B[l, x, r] := B[l, y, r] * C[y, x]
end

"""
$(TYPEDSIGNATURES)
      -- A --
         |    |
 R =  -- M -- RE
         |    |
      -- B --
"""
function update_env_right(
    RE::S, A::S, M::T, B::S, ::Val{:n}
) where {T <: AbstractArray{Float64, 4}, S <: AbstractArray{Float64, 3}}
    @tensor R[nt, nc, nb] := RE[ot, oc, ob] * A[nt, α, ot] *
                             M[nc, α, oc, β] * B[nb, β, ob] order = (ot, α, oc, β, ob)
    R
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_right(
    RE::S, A::S, M::T, B::S, ::Val{:c}
) where {T <: AbstractArray{Float64, 4}, S <: AbstractArray{Float64, 3}}
    @tensor R[nt, nc, nb] := RE[ot, oc, ob] * A[nt, α, ot] *
                             M[nc, β, oc, α] * B[nb, β, ob] order = (ot, α, oc, β, ob)
    R
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_right(
    RE::S, M::T, ::Union{Val{:n}, Val{:c}}
) where {S <: AbstractArray{Float64, 3}, T <: AbstractArray{Float64, 2}}
    @tensor R[nt, nc, nb] := M[nc, oc] * RE[nt, oc, nb]
    R
end

"""
$(TYPEDSIGNATURES)
   |    |    |
  LE -- M -- RE
   |    |    |
     -- B --
"""
function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{:n}
) where {T <: AbstractArray{Float64, 4}, S <: AbstractArray{Float64, 3}}
    @tensor A[x, y, z] := LE[k, l, x] * B[k, m, o] *
                          M[l, y, n, m] * RE[z, n, o] order = (k, l, m, n, o)
    A
end

"""
$(TYPEDSIGNATURES)
"""
function project_ket_on_bra(
    LE::S, B::S, C::S, M::T, N::T, RE::S, ::Val{:n}
) where {T <: AbstractArray{Float64, 4}, S <: AbstractArray{Float64, 3}}

    @tensor A[x, y, z, r] := LE[k, l, x] * B[k, m, o] *
                          M[l, y, n, m] * C[o, s, q] *
                          N[n, z, p, s] * RE[r, p, q] order = (k, l, m, n, o, s, p, q)
    A
end

"""
$(TYPEDSIGNATURES)
"""
function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{:n}
) where {T <: AbstractArray{Float64, 2}, S <: AbstractArray{Float64, 3}}

    @tensor A[x, y, z] := M[y, a] * B[x, a, z]
    A
end

"""
$(TYPEDSIGNATURES)
"""
function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{:c}
) where {T <: AbstractArray{Float64, 2}, S <: AbstractArray{Float64, 3}}
    @tensor A[x, y, z] := B[x, a, z] * M[a, y]
    A
end

"""
$(TYPEDSIGNATURES)
"""
function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{:c}
) where {T <: AbstractArray{Float64, 4}, S <: AbstractArray{Float64, 3}}
    @tensor A[x, y, z] := LE[k, l, x] * B[k, m, o] *
                          M[l, m, n, y] * RE[z, n, o] order = (k, l, m, n, o)
    A
end

"""
$(TYPEDSIGNATURES)
"""
function project_ket_on_bra(
    LE::S, B::S, C::S, M::T, N::T, RE::S, ::Val{:c}
) where {T <: AbstractArray{Float64, 4}, S <: AbstractArray{Float64, 3}}
    @tensor A[x, m, s, r] := LE[k, l, x] * B[k, y, o] *
                          M[l, y, n, m] * C[o, z, q] *
                          N[n, z, p, s] * RE[r, p, q] order = (k, l, y, n, o, z, p, q)
    A
end