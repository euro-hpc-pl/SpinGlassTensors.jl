export
    zipper,
    corner_matrix

struct CornerTensor{T <: Real}
    C::Tensor{T, 3}
    M::Tensor{T, 4}
    B::Tensor{T, 3}

    function CornerTensor(C, M, B)
        T = promote_type(eltype.((C, M, B))...)
        new{T}(C, M, B)
    end
end

function zipper(ψ::QMpo{R}, ϕ::QMps{R}, method::Symbol=:svd; Dcut::Int=typemax(Int), tol::Real=eps(), args...) where R <: Real
    # input ϕ should be canonized :left
    # results should be canonized :right
    D = TensorMap{R}()
    C = ones(R, 1, 1, 1)
    mpo_li = last(ψ.sites)
    for i ∈ reverse(ϕ.sites)
        while mpo_li > i
            C = contract_matrix_tensor3(ψ[mpo_li], C)
            mpo_li = left_nbrs_site(mpo_li, ψ.sites)
        end
        @assert mpo_li == i "Mismatch between QMpo and QMps sites."
        mpo_li = left_nbrs_site(mpo_li, ψ.sites)

        U, Σ, V = svd_corner_matrix(Val(method), C, ψ[i], ϕ[i], Dcut, tol, args...)

        C = U * diagm(Σ)
        s1 = size(ψ[i], 1)
        s2 = size(ψ[i], 2)
        V = permutedims(V, (2, 1))
        if i == ϕ.sites[1] V = C * V end
        @cast V[x, y, z] := V[x, (y, z)] (y ∈ 1:s2)
        @cast C[x, y, z] := C[(x, y), z] (y ∈ 1:s1)
        C = permutedims(C, (3, 2, 1))
        push!(D, i => V)
    end
    QMps(D)
end

function corner_matrix(
    C::S, M::T, B::S
) where {S <: CuArrayOrArray{R, 3}, T <: MpoTensor{R, 4}} where R <: Real
    for v ∈ reverse(M.bot) B = contract_matrix_tensor3(v, B) end
    Cnew = corner_matrix(C, M.ctr, B)
    @cast Cnew[(t1, t2), t3, t4] := Cnew[t1, t2, t3, t4]
    for v ∈ reverse(M.top) Cnew = contract_matrix_tensor3(v, Cnew) end
    @cast Cnew[t12, (t3, t4)] := Cnew[t12, t3, t4]
    Cnew
end

function svd_corner_matrix(::Val{:svd}, C, M, B, Dcut, tol, args...)
    svd(corner_matrix(C, M, B), Dcut, tol, args...)
end

function svd_corner_matrix(::Val{:psvd}, C, M, B, Dcut, tol, args...)
    psvd(corner_matrix(C, M, B), rank=Dcut)
end

function svd_corner_matrix(::Val{:psvd_sparse}, C, M, B, Dcut, tol, args...)
    psvd(CornerTensor(C, M, B), rank=Dcut)
end

# this is for psvd to work
LinearAlgebra.ishermitian(ten::CornerTensor) = false
function LinearAlgebra.mul!(y, ten::CornerTensor, x)
    x = reshape(x, 1, size(ten.M, 2), size(ten.C, 1))
    y[:] = reshape(update_env_right(ten.C, x, ten.M, ten.B), :)
end
