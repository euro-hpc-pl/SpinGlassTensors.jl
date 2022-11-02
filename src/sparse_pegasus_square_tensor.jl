"""
$(TYPEDSIGNATURES)
"""
function update_env_left(
    L::S, A::S, M::T, B::S, ::Val{:n}
) where {S <: Array{Real, 3}, T <: SparsePegasusSquareTensor}
    pr, pd = M.projs
    p1l, p2l, p1u, p2u = M.bnd_projs

    ipr = cuIdentity(eltype(L), maximum(pr))[pr, :]
    lel1, lel2, leu1, leu2 = CUDA.CuArray.(M.bnd_exp)

    A_d = permutedims(CUDA.CuArray(A), (1, 3, 2))
    L_d = permutedims(CUDA.CuArray(L), (1, 3, 2))
    B_d = permutedims(CUDA.CuArray(B), (3, 1, 2))[:, :, pd]

    @cast ll[x, y, z, l1] := L_d[x, y, z] * lel1[z, l1]
    @tensor LL[x, y, l1, l2] := ll[x, y, z, l1] * lel2[z, l2] # D x D x 2^12 x 2^6

    @cast lu[x, y, z, u1] := A_d[x, y, z] * leu1[z, u1]
    @tensor AA[x, y, u1, u2] := lu[x, y, z, u1] * leu2[z, u2]

    LL = LL[:, :, p1l, p2l]
    @cast LL[x, y, (s, r)] := LL[x, y, s, r]

    AA = AA[:, :, p1u, p2u]
    @cast AA[x, y, (s, r)] := AA[x, y, s, r]

    BB = repeat(B_d, outer=(1, 1, 1, size(pr, 1)))
    @cast BB[x, y, (z, σ)] := BB[x, y, z, σ] (σ ∈ 1:size(pr, 1))

    Lnew_no_le = BB ⊠ LL ⊠ AA  # broadcast over dims = 3

    Ln = Lnew_no_le .* reshape(CUDA.CuArray(M.loc_exp), 1, 1, :)
    @reduce Lnew[x, y, σ] := sum(z) Ln[x, y, (z, σ)] (σ ∈ 1:size(pr, 1))

    @tensor ret[x, y, r] := Lnew[x, y, z] * ipr[z, r]

    Array(permutedims(ret, (1, 3, 2)) ./ maximum(abs.(ret)))
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_left(
    L::S, A::S, M::T, B::S, ::Val{:c}
) where {S <: Array{Real, 3}, T <: SparsePegasusSquareTensor}
    pr, pd = M.projs
    p1l, p2l, p1u, p2u = M.bnd_projs

    lel1, lel2, leu1, leu2 = CUDA.CuArray.(M.bnd_exp)
    ipr = cuIdentity(eltype(L), maximum(pr))[pr, :]

    A_d = permutedims(CUDA.CuArray(A), (1, 3, 2))[:, :, pd]
    L_d = permutedims(CUDA.CuArray(L), (1, 3, 2))
    B_d = permutedims(CUDA.CuArray(B), (3, 1, 2))

    @cast ll[x, y, z, l1] := L_d[x, y, z] * lel1[z, l1]
    @tensor LL[x, y, l1, l2] := ll[x, y, z, l1] * lel2[z, l2] # D x D x 2^12 x 2^6

    @cast lu[x, y, z, u1] := B_d[x, y, z] * leu1[z, u1]
    @tensor BB[x, y, u1, u2] := lu[x, y, z, u1] * leu2[z, u2]

    LL = LL[:, :, p1l, p2l]
    @cast LL[x, y, (s, r)] := LL[x, y, s, r]

    BB = BB[:, :, p1u, p2u]
    @cast BB[x, y, (s, r)] := BB[x, y, s, r]

    AA = repeat(A_d, outer=(1, 1, 1, size(pr, 1)))
    @cast AA[x, y, (z, σ)] := AA[x, y, z, σ] (σ ∈ 1:size(pr, 1))

    Lnew_no_le = BB ⊠ LL ⊠ AA  # broadcast over dims = 3

    Ln = Lnew_no_le .* reshape(CUDA.CuArray(M.loc_exp), 1, 1, :)
    @reduce Lnew[x, y, σ] := sum(z) Ln[x, y, (z, σ)] (σ ∈ 1:size(pr, 1))

    @tensor ret[x, y, r] := Lnew[x, y, z] * ipr[z, r]

    Array(permutedims(ret, (1, 3, 2)) ./ maximum(abs.(ret)))
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_right(
    R::S, A::S, M::T, B::S, ::Val{:n}
) where {T <: SparsePegasusSquareTensor, S <: Array{Real, 3}}
    pr, pd = M.projs
    p1l, p2l, p1u, p2u = M.bnd_projs
    lel1, lel2, leu1, leu2 = CUDA.CuArray.(M.bnd_exp)

    ip1l = cuIdentity(eltype(R), maximum(p1l))[p1l, :]
    ip2l = cuIdentity(eltype(R), maximum(p2l))[p2l, :]

    A_d = permutedims(CUDA.CuArray(A), (1, 3, 2))
    R_d = permutedims(CUDA.CuArray(R), (1, 3, 2))[:, :, pr]
    B_d = permutedims(CUDA.CuArray(B), (3, 1, 2))[:, :, pd]

    @cast R_d[x, y, _, z] := R_d[x, y, z]

    @cast lu[x, y, z, l1] := A_d[x, y, z] * leu1[z, l1]
    @tensor AA[x, y, l1, l2] := lu[x, y, z, l1] * leu2[z, l2] # D x D x 2^12 x 2^6

    AA = AA[:, :, p1u, p2u]
    @cast AA[x, y, (s, r)] := AA[x, y, s, r]

    RR = repeat(R_d, outer=(1, 1, size(pd, 1), 1))
    @cast RR[x, y, (z, σ)] := RR[x, y, z, σ] (σ ∈ 1:size(pd, 1))

    BB = repeat(B_d, outer=(1, 1, 1, size(pr, 1)))
    @cast BB[x, y, (z, σ)] := BB[x, y, z, σ] (σ ∈ 1:size(pr, 1))

    Rnew_no_le = AA ⊠ RR ⊠ BB  # broadcast over dims = 3

    Rn = Rnew_no_le .* reshape(CUDA.CuArray(M.loc_exp), 1, 1, :)
    @cast Rnew[x, y, η, σ] := Rn[x, y, (η, σ)] (σ ∈ 1:size(pr, 1))

    @cast ll[x, y, z] := lel1[x, y] * lel2[x, z]
    @tensor ret[x, y, l] := Rnew[x, y, s1, s2] * ip1l[s1, l1] * ip2l[s2, l2] * ll[l, l1, l2]  order=(s2, s1, l1, l2)

    Array(permutedims(ret, (1, 3, 2)) ./ maximum(abs.(ret)))
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_right(
    R::S, A::S, M::T, B::S, ::Val{:c}
) where {T <: SparsePegasusSquareTensor, S <: Array{Real, 3}}
    pr, pd = M.projs
    p1l, p2l, p1u, p2u = M.bnd_projs
    lel1, lel2, leu1, leu2 = CUDA.CuArray.(M.bnd_exp)

    ip1l = cuIdentity(eltype(R), maximum(p1l))[p1l, :]
    ip2l = cuIdentity(eltype(R), maximum(p2l))[p2l, :]

    A_d = permutedims(CUDA.CuArray(A), (1, 3, 2))[:, :, pd]
    R_d = permutedims(CUDA.CuArray(R), (1, 3, 2))[:, :, pr]
    B_d = permutedims(CUDA.CuArray(B), (3, 1, 2))

    @cast R_d[x, y, _, z] := R_d[x, y, z]

    @cast lu[x, y, z, u1] := B_d[x, y, z] * leu1[z, u1]
    @tensor BB[x, y, u1, u2] := lu[x, y, z, u1] * leu2[z, u2] # D x D x 2^12 x 2^6

    BB = BB[:, :, p1u, p2u]
    @cast BB[x, y, (s, r)] := BB[x, y, s, r]

    RR = repeat(R_d, outer=(1, 1, size(pd, 1), 1))
    @cast RR[x, y, (z, σ)] := RR[x, y, z, σ] (σ ∈ 1:size(pd, 1))

    AA = repeat(A_d, outer=(1, 1, 1, size(pr, 1)))
    @cast AA[x, y, (z, σ)] := AA[x, y, z, σ] (σ ∈ 1:size(pr, 1))

    Rnew_no_le = AA ⊠ RR ⊠ BB  # broadcast over dims = 3

    Rn = Rnew_no_le .* reshape(CUDA.CuArray(M.loc_exp), 1, 1, :)
    @cast Rnew[x, y, η, σ] := Rn[x, y, (η, σ)] (σ ∈ 1:size(pr, 1))

    @cast ll[x, y, z] := lel1[x, y] * lel2[x, z]
    @tensor ret[x, y, l] := Rnew[x, y, s1, s2] * ip1l[s1, l1] * ip2l[s2, l2] *  ll[l, l1, l2]  order=(s2, s1, l1, l2)

    Array(permutedims(ret, (1, 3, 2)) ./ maximum(abs.(ret)))
end

function project_ket_on_bra(
    L::S, B::S, M::T, R::S, ::Val{:n}
) where {S <: Array{Real, 3}, T <: SparsePegasusSquareTensor}
    pr, pd = M.projs
    p1l, p2l, p1u, p2u = M.bnd_projs
    lel1, lel2, leu1, leu2 = CUDA.CuArray.(M.bnd_exp)

    ip1u = cuIdentity(eltype(L), maximum(p1u))[p1u, :]
    ip2u = cuIdentity(eltype(L), maximum(p2u))[p2u, :]

    L_d = permutedims(CUDA.CuArray(L), (3, 1, 2))
    B_d = permutedims(CUDA.CuArray(B), (1, 3, 2))[:, :, pd]
    R_d = permutedims(CUDA.CuArray(R), (3, 1, 2))[:, :, pr]

    @cast R_d[x, y, _, z] := R_d[x, y, z]

    @cast ll[x, y, z, l1] := L_d[x, y, z] * lel1[z, l1]
    @tensor LL[x, y, l1, l2] := ll[x, y, z, l1] * lel2[z, l2] # D x D x 2^12 x 2^6

    LL = LL[:, :, p1l, p2l]
    @cast LL[x, y, (s, r)] := LL[x, y, s, r]

    RR = repeat(R_d, outer=(1, 1, size(pd, 1), 1))
    @cast RR[x, y, (z, σ)] := RR[x, y, z, σ] (σ ∈ 1:size(pd, 1))

    BB = repeat(B_d, outer=(1, 1, 1, size(pr, 1)))
    @cast BB[x, y, (z, σ)] := BB[x, y, z, σ] (σ ∈ 1:size(pr, 1))

    Anew_no_le = LL ⊠ BB ⊠ RR

    An = Anew_no_le .* reshape(CUDA.CuArray(M.loc_exp), 1, 1, :)
    @cast Anew[x, y, η, σ] := An[x, y, (η, σ)] (σ ∈ 1:size(pr, 1))

    @cast lu[x, y, z] := leu1[x, y] * leu2[x, z]
    @tensor ret[x, y, u] := Anew[x, y, s1, s2] * ip1u[s1, u1] * ip2u[s2, u2] *  lu[u, u1, u2]  order=(s2, s1, u1, u2)

    Array(permutedims(ret, (1, 3, 2)) ./ maximum(abs.(ret)))
end

"""
$(TYPEDSIGNATURES)
"""
function project_ket_on_bra(
    L::S, B::S, M::T, R::S, ::Val{:c}
) where {S <: Array{Real, 3}, T <: SparsePegasusSquareTensor}
    pr, pd = M.projs
    p1l, p2l, p1u, p2u = M.bnd_projs
    lel1, lel2, leu1, leu2 = CUDA.CuArray.(M.bnd_exp)

    ipd = cuIdentity(eltype(L), maximum(pd))[pd, :]

    L_d = permutedims(CUDA.CuArray(L), (3, 1, 2))
    B_d = permutedims(CUDA.CuArray(B), (1, 3, 2))
    R_d = permutedims(CUDA.CuArray(R), (3, 1, 2))[:, :, pr]

    @cast R_d[x, y, _, z] := R_d[x, y, z]

    @cast ll[x, y, z, l1] := L_d[x, y, z] * lel1[z, l1]
    @tensor LL[x, y, l1, l2] := ll[x, y, z, l1] * lel2[z, l2] # D x D x 2^12 x 2^6

    @cast lu[x, y, z, u1] := B_d[x, y, z] * leu1[z, u1]
    @tensor BB[x, y, u1, u2] := lu[x, y, z, u1] * leu2[z, u2] # D x D x 2^12 x 2^6

    LL = LL[:, :, p1l, p2l]
    @cast LL[x, y, (s, r)] := LL[x, y, s, r]

    BB = BB[:, :, p1u, p2u]
    @cast BB[x, y, (s, r)] := BB[x, y, s, r]

    RR = repeat(R_d, outer=(1, 1, size(pd, 1), 1))
    @cast RR[x, y, (z, σ)] := RR[x, y, z, σ] (σ ∈ 1:size(pd, 1))

    Anew_no_le = LL ⊠ BB ⊠ RR

    An = Anew_no_le .* reshape(CUDA.CuArray(M.loc_exp), 1, 1, :)
    @reduce Anew[x, y, z] := sum(σ) An[x, y, (z, σ)] (σ ∈ 1:length(pr))

    @tensor ret[x, y, d] := Anew[x, y, z] * ipd[z, d]

    Array(permutedims(ret, (1, 3, 2)) ./ maximum(abs.(ret)))
end