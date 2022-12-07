

function CUDA.CUSPARSE.CuSparseMatrixCSC(::Type{R}, pr::Vector{Int}) where R <: Real
    n = length(pr)
    CuSparseMatrixCSC(
        CuArray(1:n+1),
        CuArray(pr),
        CUDA.ones(R, n),
        (maximum(pr), n)
    )
end

function contract_sparse_with_three(X1, X2, X3, loc_exp, p1, p2, p3, pout)
    s1, s2, _ = size(X1)
    s3, s4, _ = size(X3)
    total_memory = 2^33
    batch_size = max(Int(floor(total_memory / (8 * (s1 * s2 + s2 * s3 + s3 * s4 + s4 * s1)))), 1)

    X1, X2, X3, loc_exp = CuArray.((X1, X2, X3, loc_exp))
    F = eltype(X1)
    out = CUDA.zeros(F, maximum(pout), s1, s4)

    from, total_size = 1, length(p1)
    while from <= total_size
        to = min(total_size, from + batch_size - 1)

        X1p = X1[:, :, p1[from:to]]
        X2p = X2[:, :, p2[from:to]]
        X3p = X3[:, :, p3[from:to]]
        outp = X1p ⊠ X2p ⊠ X3p
        outp .*= reshape(loc_exp[from:to], 1, 1, :)
        @cast outp[(x, y), z] := outp[x, y, z]

        poutp = pout[from:to]
        rf, rt = minimum(poutp), maximum(poutp)
        ipr = CuSparseMatrixCSC(F, poutp .- (rf - 1))
        out[rf : rt, :, :] = out[rf : rt, :, :] .+ reshape(ipr * outp', (:, s1, s4))

        from = to + 1
    end
    Array(permutedims(out, (2, 1, 3)))
end

function update_env_left(
    LE::S, A::S, M::T, B::S, type::Union{Val{:n}, Val{:c}}
) where {S <: ArrayOrCuArray{3}, T <: SparseSiteTensor}
    B = permutedims(B, (3, 1, 2))
    LE = permutedims(LE, (1, 3, 2))
    A = permutedims(A, (1, 3, 2))
    order = type == Val{:n}() ? [4, 1, 2, 3] : [2, 1, 4, 3]
    contract_sparse_with_three(B, LE, A, M.loc_exp, M.projs[order]...)
end

function update_env_right(
    RE::S, A::S, M::SparseSiteTensor, B::S, type::Union{Val{:n}, Val{:c}}
) where S <: ArrayOrCuArray{3}
    A = permutedims(A, (1, 3, 2))
    RE = permutedims(RE, (1, 3, 2))
    B = permutedims(B, (3, 1, 2))
    order = type == Val{:n}() ? [2, 3, 4, 1] : [4, 3, 2, 1]
    contract_sparse_with_three(A, RE, B, M.loc_exp, M.projs[order]...)
end

function project_ket_on_bra(
    LE::S, B::S, M::SparseSiteTensor, RE::S, type::Union{Val{:n}, Val{:c}}
) where S <: ArrayOrCuArray{3}
    LE = permutedims(LE, (3, 1, 2))
    B = permutedims(B, (1, 3, 2))
    RE = permutedims(RE, (3, 1, 2))
    order = type == Val{:n}() ? [1, 4, 3, 2] : [1, 2, 3, 4]
    contract_sparse_with_three(LE, B, RE, M.loc_exp, M.projs[order]...)
end


function update_reduced_env_right(
    K::Array{T, 1},
    RE::Array{T, 2},
    M::SparseSiteTensor,
    B::Array{T, 3}
) where T <: Real

    B, RE, loc_exp, K = CUDA.CuArray.((B, RE, M.loc_exp, K))

    Kloc_exp = loc_exp .* K[M.projs[2]]

    B = permutedims(B, (1, 3, 2))
    RE = reshape(RE, (size(RE, 1), 1, size(RE, 2)))

    Bp = B[:, :, M.projs[4]]
    REp = RE[:, :, M.projs[3]]

    outp = dropdims(Bp ⊠ REp, dims=2)
    outp .*= reshape(Kloc_exp, 1, :)

    ipl = CuSparseMatrixCSC(eltype(B), M.projs[1])
    RRR = ipl * outp'  # TODO: is ' correct on CUDA ?
    Array(RRR')
end
