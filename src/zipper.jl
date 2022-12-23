export
    zipper,
    corner_matrix

struct CornerTensor{T <: Real}
    C::Tensor{T, 3}
    M::MpoTensor{T, 4}
    B::Tensor{T, 3}

    function CornerTensor(C, M, B)
        T = promote_type(eltype.((C, M, B))...)
        new{T}(C, M, B)
    end
end

struct ACornerTensor{T <: Real}
    C::Tensor{T, 3}
    M::MpoTensor{T, 4}
    B::Tensor{T, 3}

    function ACornerTensor(ten)
        T = eltype(ten)
        new{T}(ten.C, ten.M, ten.B)
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

        CM = CornerTensor(C, ψ[i], ϕ[i])
        U, Σ, V = svd_corner_matrix(Val(method), CM, Dcut, tol, args...)

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

function Base.Array(CM::CornerTensor)
    B, M, C = CM.B, CM.M, CM.C 
    for v ∈ reverse(M.bot) B = contract_matrix_tensor3(v, B) end
    Cnew = corner_matrix(C, M.ctr, B)
    @cast Cnew[(t1, t2), t3, t4] := Cnew[t1, t2, t3, t4]
    for v ∈ reverse(M.top) Cnew = contract_matrix_tensor3(v, Cnew) end
    @cast Cnew[t12, (t3, t4)] := Cnew[t12, t3, t4]
    Cnew
end

function svd_corner_matrix(::Val{:svd}, CM, Dcut, tol, args...)
    svd(Array(CM), Dcut, tol, args...)
end

function svd_corner_matrix(::Val{:psvd}, CM, Dcut, tol, args...)
    psvd(Array(CM), rank=Dcut)
end

function svd_corner_matrix(::Val{:psvd_sparse}, CM, Dcut, tol, args...)
    psvd(CM, rank=Dcut)
end

# this is for psvd to work
LinearAlgebra.ishermitian(ten::CornerTensor) = false
Base.size(ten::CornerTensor) = (size(ten.B, 1) * size(ten.M, 1), size(ten.C, 1) * size(ten.M, 2))
Base.size(ten::CornerTensor, n::Int) = size(ten)[n]
Base.eltype(ten::CornerTensor{T}) where T = T

Base.adjoint(ten::CornerTensor{T}) where T = ACornerTensor(ten)

function LinearAlgebra.mul!(y, ten::CornerTensor, x)
    x = reshape(x, size(ten.M, 2), size(ten.C, 1), :)
    x = permutedims(x, (3, 1, 2))
    yp = update_env_right(ten.C, x, ten.M, ten.B)
    y[:] = reshape(permutedims(yp, (3, 2, 1)), :)
end

function LinearAlgebra.mul!(y, ten::ACornerTensor, x)
    x = reshape(x, size(ten.B, 1), size(ten.M, 1), :)
    x = Array(x)
    yp = project_ket_on_bra(x, ten.B, ten.M, ten.C)
    y[:] = reshape(yp, :)
end
