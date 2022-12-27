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

struct Adjoint{T, S <: CornerTensor}
    parent::S

    function Adjoint{T}(ten::CornerTensor{S}) where {T, S}
        F = promote_type(T, S)
        new{F, CornerTensor{F}}(ten)
    end
end

"""
input ϕ (results) should be canonized :left (:right)
"""
function zipper(
    ψ::QMpo{R}, ϕ::QMps{R}; method::Symbol=:svd, Dcut::Int=typemax(Int), tol::Real=eps(), kwargs...
) where R <: Real
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
        U, Σ, V = svd_corner_matrix(CM, method, Dcut, tol, kwargs...)
        C = U * diagm(Σ)

        V = permutedims(V, (2, 1))
        if i == ϕ.sites[1] V = C * V end
        s1, s2 = size(ψ[i])
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
end

Base.Array(CM::Adjoint{T, CornerTensor{T}}) where T = adjoint(Array(CM.ten))

function svd_corner_matrix(CM, method::Symbol, Dcut::Int, tol::Real; kwargs...)
    if method == :svd
        return svd(Array(CM), Dcut, tol, kwargs...)
    elseif method == :psvd
        return psvd(Array(CM), rank=Dcut, kwargs...)
    elseif method == :psvd_sparse
        return psvd(CM, rank=Dcut, kwargs...)
    elseif method == :tsvd
        return tsvd(Array(CM), Dcut, kwargs...)
    elseif method == :tsvd_sparse
        return tsvd(CM, Dcut, kwargs...)
    else
        throw(ArgumentError("Wrong method $method"))
    end
end

# this is for psvd to work
LinearAlgebra.ishermitian(ten::CornerTensor) = false
Base.size(ten::CornerTensor) = (size(ten.B, 1) * size(ten.M, 1), size(ten.C, 1) * size(ten.M, 2))
Base.size(ten::CornerTensor, n::Int) = size(ten)[n]
Base.eltype(ten::CornerTensor{T}) where T = T
Base.adjoint(ten::CornerTensor{T}) where T = Adjoint{T}(ten)

function LinearAlgebra.mul!(y, ten::CornerTensor, x)
    x = reshape(x, size(ten.M, 2), size(ten.C, 1), :)
    x = permutedims(x, (3, 1, 2))
    yp = update_env_right(ten.C, x, ten.M, ten.B)
    y[:, :] .= reshape(permutedims(yp, (3, 2, 1)), size(ten.B, 1) * size(ten.M, 1), :)
end

function LinearAlgebra.mul!(y, ten::Adjoint{T, CornerTensor{T}}, x) where T
    x = reshape(x, size(ten.parent.B, 1), size(ten.parent.M, 1), :)
    x = Array(x) # TODO This could be avoided
    yp = project_ket_on_bra(x, ten.parent.B, ten.parent.M, ten.parent.C)
    y[:, :] .= reshape(permutedims(yp, (2, 3, 1)), size(ten.parent.M, 2) * size(ten.parent.C, 1), :)
end

function Base.:(*)(ten::CornerTensor{T}, x::Vector{T}) where T
    x = reshape(x, size(ten.M, 2), size(ten.C, 1), 1)
    x = permutedims(x, (3, 1, 2))
    yp = update_env_right(ten.C, x, ten.M, ten.B)
    reshape(permutedims(yp, (3, 2, 1)), size(ten.B, 1) * size(ten.M, 1))
end

function Base.:(*)(ten::Adjoint{T, CornerTensor{T}}, x::Vector{T}) where T
    x = reshape(x, size(ten.ten.B, 1), size(ten.ten.M, 1), 1)
    yp = project_ket_on_bra(x, ten.ten.B, ten.ten.M, ten.ten.C)
    reshape(permutedims(yp, (2, 3, 1)), size(ten.ten.M, 2) * size(ten.ten.C, 1))
end
