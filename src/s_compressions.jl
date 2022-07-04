export
    variational_compress!,
    _left_nbrs_site,
    _right_nbrs_site,
    compress_twosite!,
    canonise_truncate!,
    variational_sweep!,
    Environment,
    optimize_gauges_for_overlaps!!

"""
$(TYPEDSIGNATURES)
"""
mutable struct Environment <: AbstractEnvironment
    bra::QMps  # to be optimized
    mpo::QMpo
    ket::QMps
    trans::Symbol
    env::Dict

    function Environment(
        bra::QMps,
        mpo::QMpo,
        ket::QMps,
        trans::Symbol=:n
    )
        @assert trans ∈ (:n, :c)
        @assert bra.sites == ket.sites
        @assert issubset(bra.sites, mpo.sites)

        env0 = Dict(
            (first(bra.sites), :left) => ones(1, 1, 1),
            (last(bra.sites), :right) => ones(1, 1, 1)
        )
        env = new(bra, mpo, ket, trans, env0)
        for site ∈ env.bra.sites update_env_left!(env, site, trans) end
        env
    end
end

"""
$(TYPEDSIGNATURES)
"""
function variational_compress!(
    bra::QMps,
    mpo::QMpo,
    ket::QMps,
    tol::Real=1E-10,
    max_sweeps::Int=4,
    trans::Symbol=:n,
    args...
)
    env = Environment(bra, mpo, ket, trans)
    overlap = Inf
    overlap_before = measure_env(env, last(env.bra.sites), trans)

    for sweep ∈ 1:max_sweeps
        _left_sweep_var!(env, trans, args...)
        _right_sweep_var!(env, trans, args...)

        overlap = measure_env(env, last(env.bra.sites), trans)
        Δ = abs(overlap_before - abs(overlap))
        @info "Convergence" Δ

        if Δ < tol
            @info "Finished in $sweep sweeps of $(max_sweeps)."
            return overlap
        else
            overlap_before = overlap
        end
    end
    overlap
end

"""
$(TYPEDSIGNATURES)
"""
function _left_sweep_var!(env::Environment, trans::Symbol=:n, args...)
    for site ∈ reverse(env.bra.sites)
        update_env_right!(env, site, trans)
        A = project_ket_on_bra(env, site, trans)
        @cast B[x, (y, z)] := A[x, y, z]
        _, Q = rq_fact(B, args...)
        @cast C[x, σ, y] := Q[x, (σ, y)] (σ ∈ 1:size(A, 2))
        env.bra[site] = C
        clear_env_containing_site!(env, site)
    end
end

"""
$(TYPEDSIGNATURES)
"""
function _right_sweep_var!(env::Environment, trans::Symbol=:n, args...)
    for site ∈ env.bra.sites
        update_env_left!(env, site, trans)
        A = project_ket_on_bra(env, site, trans)
        @cast B[(x, y), z] := A[x, y, z]
        Q, _ = qr_fact(B, args...)
        @cast C[x, σ, y] := Q[(x, σ), y] (σ ∈ 1:size(A, 2))
        env.bra[site] = C
        clear_env_containing_site!(env, site)
    end
end

"""
$(TYPEDSIGNATURES)
Largest x in sites: x < site
"""
function _left_nbrs_site(site::Site, sites)
    ls = filter(i -> i < site, sites)
    if isempty(ls) return -Inf end
    maximum(ls)
end

"""
$(TYPEDSIGNATURES)
Smallest x in sites: x > site
"""
function _right_nbrs_site(site::Site, sites)
    ms = filter(i -> i > site, sites)
    if isempty(ms) return Inf end
    minimum(ms)
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_left!(env::Environment, site::Site, trans::Symbol=:n)
    if site <= first(env.bra.sites) return end

    ls = _left_nbrs_site(site, env.bra.sites)
    LL = update_env_left(env.env[(ls, :left)], env.bra[ls], env.mpo[ls], env.ket[ls], trans)

    rs = _right_nbrs_site(ls, env.mpo.sites)
    while rs < site
        LL = update_env_left(LL, env.mpo[rs], Val(trans))
        rs = _right_nbrs_site(rs, env.mpo.sites)
    end
    push!(env.env, (site, :left) => LL)
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_right!(env::Environment, site::Site, trans::Symbol=:n)
    if site >= last(env.bra.sites) return end

    rs = _right_nbrs_site(site, env.bra.sites)
    RR = update_env_right(env.env[(rs, :right)], env.bra[rs], env.mpo[rs], env.ket[rs], trans)

    ls = _left_nbrs_site(rs, env.mpo.sites)
    while ls > site
        RR = update_env_right(RR, env.mpo[ls], Val(trans))
        ls = _left_nbrs_site(ls, env.mpo.sites)
    end
    push!(env.env, (site, :right) => RR)
end

"""
$(TYPEDSIGNATURES)
"""
function clear_env_containing_site!(env::Environment, site::Site)
    delete!(env.env, (_left_nbrs_site(site, env.ket.sites), :right))
    delete!(env.env, (_right_nbrs_site(site, env.ket.sites), :left))
end

"""
$(TYPEDSIGNATURES)
        -- A --
      |    |
 L = LE -- M --
      |    |
        -- B --
"""
function update_env_left(
    LE::S, A₀::S, M::T, B₀::S, trans::Symbol=:n
) where {S <: AbstractArray{Float64, 3}, T <: AbstractDict}
    sites = sort(collect(keys(M)))
    A = _update_tensor_forward(A₀, M, sites, Val(trans))
    B = _update_tensor_backwards(B₀, M, sites, Val(trans))
    update_env_left(LE, A, M[0], B, Val(trans))
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_left(
    LE::S, M::T, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: AbstractDict}
    MM = M[0]  # Can be more general
    @tensor L[nt, nc, nb] :=  LE[nt, oc, nb] * MM[oc, nc]
    L
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_left(
    LE::S, M::T, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: AbstractDict}
    MM = M[0]  # Can be more general
    @tensor L[nt, nc, nb] :=  LE[nt, oc, nb] * MM[oc, nc]
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

#=
#TODO: experimental ((uses ⊠ operator - batched_multiply - from NNlib / NNlibCUDA)
# Remove CUDA.CuArray to use CPU
"""
$(TYPEDSIGNATURES)
"""
function update_env_left(
    LE::S, A::S, M::T, B::S, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: SparseSiteTensor}
    L = zeros(size(B, 3), maximum(M.projs[3]), size(A, 3))

    A_d = permutedims(CUDA.CuArray(A[:, M.projs[2], :]), (1, 3, 2))
    L_d = permutedims(CUDA.CuArray(LE[:, M.projs[1], :]), (1, 3, 2))
    B_d = permutedims(CUDA.CuArray(B[:, M.projs[4], :]), (3, 1, 2))

    Lr_d = B_d ⊠ L_d ⊠ A_d
    Lr_d .*= reshape(CUDA.CuArray(M.loc_exp), 1, 1, :)

    Lr = Array(Lr_d)
    Threads.@threads for r ∈ 1:maximum(M.projs[3])
        σ = findall(M.projs[3] .== r)
        L[:, r, :] = sum(Lr[:, :, σ], dims=3)
    end
    L
end
=#

#TODO: This implementation may not be optimal as is not batching matrix multiplication.
"""
$(TYPEDSIGNATURES)
"""
function update_env_left(
     LE::S, A::S, M::T, B::S, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: SparseSiteTensor}
    Ap = permutedims(A, (1, 3, 2))
    LEp = permutedims(LE, (1, 3, 2))
    Bp = permutedims(B, (3, 1, 2))

    L = zeros(size(B, 3), size(A, 3), maximum(M.projs[3]))
    for (σ, lexp) ∈ enumerate(M.loc_exp)
        AA = @inbounds @view Ap[:, :, M.projs[2][σ]]
        LL = @inbounds @view LEp[:, :, M.projs[1][σ]]
        BB = @inbounds @view Bp[:, :, M.projs[4][σ]]
        @inbounds L[:, :, M.projs[3][σ]] += lexp .* (BB * LL * AA)
     end
     permutedims(L, (1, 3, 2))
end


"""
$(TYPEDSIGNATURES)
"""
function update_env_left(
    L::S, A::S, M::T, B::S, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: SparsePegasusSquareTensor}
@time begin
    println("Pre-processing ...")
    _, _, pr, pd = M.projs
    p1l, p2l, p1u, p2u = M.bnd_projs

    le1l, le2l, le1u, le2u = M.bnd_exp

    le1l = CUDA.CuArray(le1l')
    le2l = CUDA.CuArray(le2l')
    le1u = CUDA.CuArray(le1u')
    le2u = CUDA.CuArray(le2u')

    loc_exp = CUDA.CuArray(M.loc_exp')

    en1, en2 = M.loc_en

    A_d = permutedims(CUDA.CuArray(A), (1, 3, 2))
    L_d = permutedims(CUDA.CuArray(L), (1, 3, 2))
    B_d = permutedims(CUDA.CuArray(B), (3, 1, 2))

    newL = CUDA.zeros(Float64, size(B, 3), size(A, 3), maximum(pr))

    le1l = view(le1l, :, p1l)
    le1u = view(le1u, :, p1u)
    BB = view(B_d, :, :, pd)

end
#=
println("Starting contraction ...")
@time begin
    for s2 ∈ 1:length(en2)
        ll = le1l .* view(le2l, :, p2l[s2])
        lu = le1u .* view(le2u, :, p2u[s2])
println("a")
        @matmul AA[d, y, b] := sum(x) A_d[x, (d, y)] * lu[x, b] (d ∈ 1:size(A, 1))
println("b")
        @matmul LL[d, y, b] := sum(x) L_d[x, (d, y)] * ll[x, b] (d ∈ 1:size(LE, 1))
println("c")
        #LnoMloc = BB ⊠ LL ⊠ AA
        LnoMloc = CUDA.zeros(size(BB, 1), size(AA, 2), size(le1u, 1))
        for k ∈ 1:size(AA, 3)
            LnoMloc[:, :, k] = view(BB, :, :, k) * view(LL, :, :, k) * view(AA, :, :, k)
        end
println("d")
        Mloc = reshape(view(loc_exp, :, s2), 1, 1, length(en1))
println("e")
        L[:, :, pr[s2]] += reduce(+, Mloc .* LnoMloc; dims=3)
    end
end
    Array(permutedims(L, (1, 3, 2)) ./ maximum(abs.(L)))
=#
        #@time begin
        ll = lel1 .* view(lel2, :, p2l[s2])
        lu = leu1 .* view(leu2, :, p2u[s2])
        @matmul AA[x, y, s1] := sum(z) A_d[x, y, z] * lu[z, s1]
        @matmul LL[x, y, s1] := sum(z) L_d[x, y, z] * ll[z, s1]
        L_no_le = BB ⊠ LL ⊠ AA  # broadcast over dims = 3
        le_s = view(loc_exp, :, s2)
        @matmul LL_s2[x, y] := sum(s1) L_no_le[x, y, s1] * le_s[s1]
        newL[:, :, pr[s2]] += LL_s2
        #end
    end
    Array(permutedims(newL, (1, 3, 2)) ./ maximum(abs.(newL)))
end

#=
function _update_env_left_kernel(
    en1_size, en2_size,
    p1l, p2l, p1u, p2u,
    le1l, le2l, le1u, le2u,
    pr, pd,
    AA, LL,
    A, B, LE,
    loc_exp,
    ret
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    x_stride = gridDim().x * blockDim().x

    idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    y_stride = gridDim().y * blockDim().y

    for s1 ∈ idx:x_stride:en1_size, s2 ∈ idy:y_stride:en2_size
        for d ∈ 1:size(AA, 1), y ∈ 1:size(AA, 2), k ∈ 1:size(le1l, 1)
            @inbounds AA[d, y] += le1l[k, p1l[s1]] * le2l[k, p2l[s2]] * A[k, d, y]
        end

        for d ∈ 1:size(LL, 1), y ∈ 1:size(LL, 2), k ∈ 1:size(le1u, 1)
            @inbounds LL[d, y] += le1u[k, p1u[s1]] * le2u[k, p2u[s2]] * LE[k, d, y]
        end

        α = loc_exp[s2, s1]
        for i ∈ 1:size(ret, 1), j ∈ 1:size(ret, 2), l ∈ 1:size(LL, 2), k ∈ 1:size(LL, 1)
            @inbounds ret[i, j, pr[s2]] += α * B[i, k, pd[s1]] * LL[k, l] * AA[l, j]
        end
    end
end

function update_env_left(
     LE::S, A::S, M::T, B::S, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: SparsePegasusSquareTensor}
@time begin
    println("Pre-processing starts ...")

    _, _, pr, pd = M.projs
    pr, pd = CUDA.CuArray(pr), CUDA.CuArray(pd)

    p1l, p2l, p1u, p2u = M.bnd_projs
    p1l, p2l = CUDA.CuArray(p1l), CUDA.CuArray(p2l)
    p1u, p2u = CUDA.CuArray(p1u), CUDA.CuArray(p2u)

    le1l, le2l, le1u, le2u = M.bnd_exp
    le1l = CUDA.CuArray(le1l')
    le2l = CUDA.CuArray(le2l')
    le1u = CUDA.CuArray(le1u')
    le2u = CUDA.CuArray(le2u')

    en1, en2 = M.loc_en

    A_d = CUDA.CuArray(permutedims(A, (2, 1, 3)))
    LE_d = CUDA.CuArray(permutedims(LE, (2, 1, 3)))
    B_d = CUDA.CuArray(permutedims(B, (3, 1, 2)))
    loc_exp = CUDA.CuArray(M.loc_exp)

    AA = CUDA.zeros(size(A, 2), size(A, 3))
    LL = CUDA.zeros(size(LE, 2), size(LE, 3))
    L = CUDA.zeros(size(B, 3), size(A, 3), maximum(pr))

    M, N = length(en1), length(en2)
    th = (16, 16)
    bl = (ceil(Int, M / th[1]), ceil(Int, N / th[2]))

end
    @time begin
        println("Kernel starts ...")

        CUDA.@sync begin
            @cuda threads=th blocks=bl _update_env_left_kernel(
                M, N,
                p1l, p2l, p1u, p2u,
                le1l, le2l, le1u, le2u,
                pr, pd,
                AA, LL,
                A_d, B_d, LE_d,
                loc_exp,
                L
            )
        end
        println("Kernel ends.")
    end
    Array(permutedims(L, (1, 3, 2)) ./ maximum(abs.(L)))
 end
=#
"""
$(TYPEDSIGNATURES)
"""
function update_env_left(
    LE::S, A::S, M::T, B::S, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: SparseSiteTensor}
    L = zeros(size(B, 3), maximum(M.projs[3]), size(A, 3))
    for (σ, lexp) ∈ enumerate(M.loc_exp)
        AA = @inbounds @view A[:, M.projs[4][σ], :]
        LL = @inbounds @view LE[:, M.projs[1][σ], :]
        BB = @inbounds @view B[:, M.projs[2][σ], :]
        @inbounds L[:, M.projs[3][σ], :] += lexp .* (BB' * LL * AA)
    end
    L ./ maximum(abs.(L))
end

#=
function update_env_left(
    L::S, A::S, M::T, B::S, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: SparsePegasusSquareTensor}
    _, _, pr, pd = M.projs
    p1l, p2l, p1u, p2u = M.bnd_projs

    le1l, le2l, le1u, le2u = M.bnd_exp
    lel1 = CUDA.CuArray(le1l')
    lel2 = CUDA.CuArray(le2l')
    leu1 = CUDA.CuArray(le1u')
    leu2 = CUDA.CuArray(le2u')

    loc_exp = CUDA.CuArray(M.loc_exp')

    en1, en2 = M.loc_en

    A_d = permutedims(CUDA.CuArray(A), (1, 3, 2))
    L_d = permutedims(CUDA.CuArray(L), (1, 3, 2))
    B_d = permutedims(CUDA.CuArray(B), (3, 1, 2))

    newL = CUDA.zeros(Float64, size(B, 3), size(A, 3), maximum(pr))

    lel1 = lel1[:, p1l]
    leu1 = leu1[:, p1u]
    AA = A_d[:, :, pd]

    for s2 ∈ 1:length(en2)
        ll = lel1 .* view(lel2, :, p2l[s2])
        lu = leu1 .* view(leu2, :, p2u[s2])
        @matmul BB[x, y, s1] := sum(z) B_d[x, y, z] * lu[z, s1]
        @matmul LL[x, y, s1] := sum(z) L_d[x, y, z] * ll[z, s1]
        L_no_le = BB ⊠ LL ⊠ AA  # broadcast over dims = 3
        le_s = view(loc_exp, :, s2)
        @matmul LL_s2[x, y] := sum(s1) L_no_le[x, y, s1] * le_s[s1]
        newL[:, :, pr[s2]] += LL_s2
    end
    Array(permutedims(newL, (1, 3, 2)) ./ maximum(abs.(newL)))
end
=#

# function update_env_left(
#     LE::S, A::S, M::T, B::S, ::Val{:c}
# ) where {S <: AbstractArray{Float64, 3}, T <: SparsePegasusSquareTensor}
#     pl, pu, pr, pd = M.projs
#     le1l, le2l, le1u, le2u = M.bnd_exp
#     p1l, p2l, p1u, p2u = M.bnd_projs
#     en1, en2 = M.loc_en
#     sB1, sB2, sB3 = size(B)
#     sL1, sL2, sL3 = size(LE)
#     L = zeros(size(B, 3), size(A, 3), maximum(pr))
#     B_d = reshape(permutedims(B, (2, 3, 1)), sB2, sB3 * sB1)
#     LE_d = reshape(permutedims(LE, (2, 1, 3)), sL2, sL1 * sL3)
#     A_d = permutedims(A, (1, 3, 2))
#     for s1 ∈ 1:length(en1), s2 ∈ 1:length(en2)
#         ll = le1l[p1l[s1], :] .* le2l[p2l[s2], :]
#         lu = le1u[p1u[s1], :] .* le2u[p2u[s2], :]
#         BB = reshape(lu' * B_d, sB3, sB1)
#         LL = reshape(ll' * LE_d, sL1, sL3)
#         AA = @view A_d[:, :, pd[s1]]
#         L[:, :, pr[s2]] += M.loc_exp[s2, s1] .* (BB * LL * AA)
#     end
#     permutedims(L, (1, 3, 2)) ./ maximum(abs.(L))
# end


"""
$(TYPEDSIGNATURES)
"""
function update_env_left(
    LE::S, A::S, M::T, B::S, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    @cast A4[x, k, l, y] := A[x, (k, l), y] (k ∈ 1:maximum(p_rt))
    @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lb))

    L = zeros(size(B, 3), length(p_r), size(A, 3))
    for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
        AA = @inbounds @view A4[:, p_rt[r], p_lt[l], :]
        LL = @inbounds @view LE[:, l, :]
        BB = @inbounds @view B4[:, p_lb[l], p_rb[r], :]
        @inbounds L[:, r, :] += h[p_l[l], p_r[r]] .* (BB' * LL * AA)
    end
    L
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_left(
    LE::S, A::S, M::T, B::S, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs

    @cast A4[x, k, l, y] := A[x, (k, l), y] (k ∈ 1:maximum(p_rt))
    @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lb))

    L = zeros(size(B, 3), length(p_r), size(A, 3))
    for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
        AA = @inbounds @view A4[:, p_lb[l], p_rb[r], :]
        LL = @inbounds @view LE[:, l, :]
        BB = @inbounds @view B4[:, p_rt[r], p_lt[l], :]
        @inbounds L[:, r, :] += h[p_l[l], p_r[r]] .* (BB' * LL * AA)
    end
    L
end

"""
$(TYPEDSIGNATURES)
"""
function _update_tensor_forward(
    A::S, M::T, sites, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: AbstractDict}
    B = copy(A)
    for i ∈ sites
        if i == 0 break end
        C = M[i]
        @tensor B[l, x, r] := B[l, y, r] * C[y, x]
    end
    B
end

"""
$(TYPEDSIGNATURES)
"""
function _update_tensor_forward(
    A::S, M::T, sites, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: AbstractDict}
    B = copy(A)
    for i ∈ reverse(sites)
        if i == 0 break end
        C = M[i]
        @tensor B[l, x, r] := B[l, y, r] * C[x, y]
    end
    B
end

"""
$(TYPEDSIGNATURES)
"""
function _update_tensor_backwards(
    A::S, M::T, sites, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: AbstractDict}
    B = copy(A)
    for i ∈ reverse(sites)
        if i == 0 break end
        C = M[i]
        @tensor B[l, x, r] := B[l, y, r] * C[x, y]
    end
    B
end

"""
$(TYPEDSIGNATURES)
"""
function _update_tensor_backwards(
    A::S, M::T, sites, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: AbstractDict}
    B = copy(A)
    for i ∈ sites
        if i == 0 break end
        C = M[i]
        @tensor B[l, x, r] := B[l, y, r] * C[y, x]
    end
    B
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
    RE::S, A::S, M::T, B::S, ::Val{:n}
) where {T <: SparseSiteTensor, S <: AbstractArray{Float64, 3}}
    R = zeros(size(A, 1), maximum(M.projs[1]), size(B, 1))

    for (σ, lexp) ∈ enumerate(M.loc_exp)
        AA = @inbounds @view A[:, M.projs[2][σ], :]
        RR = @inbounds @view RE[:, M.projs[3][σ], :]
        BB = @inbounds @view B[:, M.projs[4][σ], :]
        @inbounds R[:, M.projs[1][σ], :] += lexp .* (AA * RR * BB')
    end
    R
end


"""
$(TYPEDSIGNATURES)
"""
function update_env_right(
    R::S, A::S, M::T, B::S, ::Val{:n}
) where {T <: SparsePegasusSquareTensor, S <: AbstractArray{Float64, 3}}
    pl, pu, pr, pd = M.projs
    p1l, p2l, p1u, p2u = M.bnd_projs

    le1l, le2l, le1u, le2u = M.bnd_exp
    lel1 = CUDA.CuArray(le1l')
    lel2 = CUDA.CuArray(le2l')
    leu1 = CUDA.CuArray(le1u')
    leu2 = CUDA.CuArray(le2u')

    loc_exp = CUDA.CuArray(M.loc_exp')  # [s1 s2]

    en1, en2 = M.loc_en

    A_d = permutedims(CUDA.CuArray(A), (1, 3, 2))
    R_d = permutedims(CUDA.CuArray(R), (1, 3, 2))
    B_d = permutedims(CUDA.CuArray(B), (3, 1, 2))

    newR = CUDA.zeros(Float64, size(A, 1), size(B, 1), maximum(pl))

    lel1 = lel1[:, p1l]
    leu1 = leu1[:, p1u]
    BB = B_d[:, :, pd]

    for s2 ∈ 1:length(en2)
        ll = lel1 .* view(lel2, :, p2l[s2])
        lu = leu1 .* view(leu2, :, p2u[s2])
        @matmul AA[x, y, s1] := sum(z) A_d[x, y, z] * lu[z, s1]
        RR = view(R_d, :, :, pr[s2])
        R_no_le = AA ⊠ RR ⊠ BB  # broadcast over dims = 3
        le_s = view(loc_exp, :, s2)
        R_no_le = R_no_le .* reshape(le_s, 1, 1, :)
        @matmul RR_s2[x, y, z] := sum(s1) R_no_le[x, y, s1] * ll[z, s1]
        newR[:, :, :] += RR_s2
    end
    Array(permutedims(newR, (1, 3, 2)) ./ maximum(abs.(newR)))
end



# """
# $(TYPEDSIGNATURES)
# """
# function update_env_right(
#     RE::S, A::S, M::T, B::S, ::Val{:n}
# ) where {T <: SparsePegasusSquareTensor, S <: AbstractArray{Float64, 3}}
#     pl, pu, pr, pd = M.projs
#     le1l, le2l, le1u, le2u = M.bnd_exp
#     p1l, p2l, p1u, p2u = M.bnd_projs
#     en1, en2 = M.loc_en
#     R = zeros(size(A, 1), maximum(pl), size(B, 1))

#     for s1 ∈ 1:length(en1), s2 ∈ 1:length(en2)
#         lu = le1u[p1u[s1], :] .* le2u[p2u[s2], :]
#         @tensor AA[x, y] := A[x, z, y] * lu[z]
#         RR = @view RE[:, pr[s2], :]
#         BB = @view B[:, pd[s1], :]
#         ll = reshape(le1l[p1l[s1], :] .* le2l[p2l[s2], :], 1, :, 1)
#         sA = size(AA, 1)
#         sB = size(BB, 1)
#         Rpart = reshape(AA * RR * BB', sA, 1, sB)
#         R[:, :, :] += M.loc_exp[s2, s1] .* (Rpart .* ll)
#     end
#     R ./ maximum(abs.(R))
# end

"""
$(TYPEDSIGNATURES)
"""
function update_env_right(
    RE::S, A::S, M::T, B::S, ::Val{:c}
) where {T <: SparseSiteTensor, S <: AbstractArray{Float64, 3}}
    R = zeros(size(A, 1), maximum(M.projs[1]), size(B, 1))

    for (σ, lexp) ∈ enumerate(M.loc_exp)
        AA = @inbounds @view A[:, M.projs[4][σ], :]
        RR = @inbounds @view RE[:, M.projs[3][σ], :]
        BB = @inbounds @view B[:, M.projs[2][σ], :]
        @inbounds R[:, M.projs[1][σ], :] += lexp .* (AA * RR * BB')
    end
    R
end


"""
$(TYPEDSIGNATURES)
"""
function update_env_right(
    R::S, A::S, M::T, B::S, ::Val{:c}
) where {T <: SparsePegasusSquareTensor, S <: AbstractArray{Float64, 3}}
    pl, pu, pr, pd = M.projs
    p1l, p2l, p1u, p2u = M.bnd_projs

    le1l, le2l, le1u, le2u = M.bnd_exp
    lel1 = CUDA.CuArray(le1l')
    lel2 = CUDA.CuArray(le2l')
    leu1 = CUDA.CuArray(le1u')
    leu2 = CUDA.CuArray(le2u')

    loc_exp = CUDA.CuArray(M.loc_exp')  # [s1 s2]

    en1, en2 = M.loc_en

    A_d = permutedims(CUDA.CuArray(A), (1, 3, 2))
    R_d = permutedims(CUDA.CuArray(R), (1, 3, 2))
    B_d = permutedims(CUDA.CuArray(B), (3, 1, 2))

    newR = CUDA.zeros(Float64, size(A, 1), size(B, 1), maximum(pl))

    lel1 = lel1[:, p1l]
    leu1 = leu1[:, p1u]
    AA = A_d[:, :, pd]

    for s2 ∈ 1:length(en2)
        ll = lel1 .* view(lel2, :, p2l[s2])
        lu = leu1 .* view(leu2, :, p2u[s2])
        @matmul BB[x, y, s1] := sum(z) B_d[x, y, z] * lu[z, s1]
        RR = view(R_d, :, :, pr[s2])
        R_no_le = AA ⊠ RR ⊠ BB  # broadcast over dims = 3
        le_s = view(loc_exp, :, s2)
        R_no_le = R_no_le .* reshape(le_s, 1, 1, :)
        @matmul RR_s2[x, y, z] := sum(s1) R_no_le[x, y, s1] * ll[z, s1]
        newR[:, :, :] += RR_s2
    end
    Array(permutedims(newR, (1, 3, 2)) ./ maximum(abs.(newR)))
end


# """
# $(TYPEDSIGNATURES)
# """
# function update_env_right(
#     RE::S, A::S, M::T, B::S, ::Val{:c}
# ) where {T <: SparsePegasusSquareTensor, S <: AbstractArray{Float64, 3}}
#     pl, pu, pr, pd = M.projs
#     le1l, le2l, le1u, le2u = M.bnd_exp
#     p1l, p2l, p1u, p2u = M.bnd_projs
#     en1, en2 = M.loc_en
#     R = zeros(size(A, 1), maximum(pl), size(B, 1))

#     for s1 ∈ 1:length(en1), s2 ∈ 1:length(en2)
#         lu = le1u[p1u[s1], :] .* le2u[p2u[s2], :]
#         @tensor BB[x, y] := B[x, z, y] * lu[z]
#         RR = @view RE[:, pr[s2], :]
#         AA = @view A[:, pd[s1], :]
#         ll = reshape(le1l[p1l[s1], :] .* le2l[p2l[s2], :], 1, :, 1)
#         sA = size(AA, 1)
#         sB = size(BB, 1)
#         Rpart = reshape(AA * RR * BB', sA, 1, sB)
#         R[:, :, :] += M.loc_exp[s2, s1] .* (Rpart .* ll)
#     end
#     R ./ maximum(abs.(R))
# end


"""
$(TYPEDSIGNATURES)
"""
function update_env_right(
    RE::S, A::S, M::T, B::S, ::Val{:n}
) where {T <: SparseVirtualTensor, S <: AbstractArray{Float64,3}}
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs
    @cast A4[x, k, l, y] := A[x, (k, l), y] (k ∈ 1:maximum(p_rt))
    @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lb))

    R = zeros(size(A, 1), length(p_l), size(B, 1))
    for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
        AA = @inbounds @view A4[:, p_rt[r], p_lt[l], :]
        RR = @inbounds @view RE[:, r, :]
        BB = @inbounds @view B4[:, p_lb[l], p_rb[r], :]
        @inbounds R[:, l, :] += h[p_l[l], p_r[r]] * (AA * RR * BB')
    end
    R
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_right(
    RE::S, A::S, M::T, B::S, ::Val{:c}
) where {T <: SparseVirtualTensor, S <: AbstractArray{Float64, 3}}
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs
    @cast A4[x, k, l, y] := A[x, (k, l), y] (k ∈ 1:maximum(p_rt))
    @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lb))

    R = zeros(size(A, 1), length(p_l), size(B, 1))
    for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
        AA = @inbounds @view A4[:, p_lb[l], p_rb[r], :]
        RR = @inbounds @view RE[:, r, :]
        BB = @inbounds @view B4[:, p_rt[r], p_lt[l], :]
        @inbounds R[:, l, :] += h[p_l[l], p_r[r]] * (AA * RR * BB')
    end
    R
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_right(
    RE::S, A₀::S1, M::T, B₀::S, trans::Symbol=:n
) where {T <: AbstractDict, S <: AbstractArray{Float64, 3}, S1 <: AbstractArray{Float64, 3}}
    sites = sort(collect(keys(M)))
    A = _update_tensor_forward(A₀, M, sites, Val(trans))
    B = _update_tensor_backwards(B₀, M, sites, Val(trans))
    update_env_right(RE, A, M[0], B, Val(trans))
end

"""
$(TYPEDSIGNATURES)
           --
              |
 R =  -- M -- RE
              |
           --
"""
function update_env_right(
    RE::S, M::T, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: AbstractDict}
    MM = M[0]
    @tensor R[nt, nc, nb] := MM[nc, oc] * RE[nt, oc, nb]
    R
end

"""
$(TYPEDSIGNATURES)
"""
function update_env_right(
    RE::S, M::T, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: AbstractDict}
    MM = M[0]
    @tensor R[nt, nc, nb] := MM[nc, oc] * RE[nt, oc, nb]
    R
end

"""
$(TYPEDSIGNATURES)
"""
function project_ket_on_bra(env::Environment, site::Site, trans::Symbol=:n)
    project_ket_on_bra(
        env.env[(site, :left)],
        env.ket[site],
        env.mpo[site],
        env.env[(site, :right)],
        Val(trans)
    )
end

"""
$(TYPEDSIGNATURES)
"""
function project_ket_on_bra_twosite(env::Environment, site::Site)
    site_l = _left_nbrs_site(site, env.bra.sites)
    project_ket_on_bra(
        env.env[(site_l, :left)],
        env.ket[site_l],
        env.ket[site],
        env.mpo[site_l][0],
        env.mpo[site][0],
        env.env[(site, :right)]
    )
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
    @tensor A[x, y, z] := B[x, a, z] * M[y, a]
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
    LE::S, B::S, M::T, RE::S, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: SparseSiteTensor}
    A = zeros(size(LE, 3), maximum(M.projs[2]), size(RE, 1))

    for (σ, lexp) ∈ enumerate(M.loc_exp)
        le = @inbounds @view LE[:, M.projs[1][σ], :]
        b = @inbounds @view B[:, M.projs[4][σ], :]
        re = @inbounds @view RE[:, M.projs[3][σ], :]
        @inbounds A[:, M.projs[2][σ], :] += lexp .* (le' * b * re')
    end
    A
end



function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: SparsePegasusSquareTensor}
    pl, pu, pr, pd = M.projs
    p1l, p2l, p1u, p2u = M.bnd_projs

    le1l, le2l, le1u, le2u = M.bnd_exp
    lel1 = CUDA.CuArray(le1l')
    lel2 = CUDA.CuArray(le2l')
    leu1 = CUDA.CuArray(le1u')
    leu2 = CUDA.CuArray(le2u')

    loc_exp = CUDA.CuArray(M.loc_exp')  # [s1 s2]

    en1, en2 = M.loc_en

    L_d = permutedims(CUDA.CuArray(LE), (3, 1, 2))
    R_d = permutedims(CUDA.CuArray(RE), (3, 1, 2))
    B_d = permutedims(CUDA.CuArray(B), (1, 3, 2))

    newA = CUDA.zeros(Float64, size(LE, 3), size(RE, 1), maximum(pu))

    lel1 = lel1[:, p1l]
    leu1 = leu1[:, p1u]
    BB = B_d[:, :, pd]

    for s2 ∈ 1:length(en2)
        ll = lel1 .* view(lel2, :, p2l[s2])
        lu = leu1 .* view(leu2, :, p2u[s2])
        @matmul LL[x, y, s1] := sum(z) L_d[x, y, z] * ll[z, s1]
        RR = view(R_d, :, :, pr[s2])
        A_no_le = LL ⊠ BB ⊠ RR  # broadcast over dims = 3
        le_s = view(loc_exp, :, s2)
        A_no_le = A_no_le .* reshape(le_s, 1, 1, :)
        @matmul AA_s2[x, y, z] := sum(s1) A_no_le[x, y, s1] * lu[z, s1]
        newA[:, :, :] += AA_s2
    end
    Array(permutedims(newA, (1, 3, 2)) ./ maximum(abs.(newA)))
end


# """
# $(TYPEDSIGNATURES)
# """
# function project_ket_on_bra(
#     LE::S, B::S, M::T, RE::S, ::Val{:n}
# ) where {S <: AbstractArray{Float64, 3}, T <: SparsePegasusSquareTensor}
#     pl, pu, pr, pd = M.projs
#     le1l, le2l, le1u, le2u = M.bnd_exp
#     p1l, p2l, p1u, p2u = M.bnd_projs
#     en1, en2 = M.loc_en
#     L = zeros(size(LE, 3), maximum(pu), size(RE, 1))
#     for s1 ∈ 1:length(en1), s2 ∈ 1:length(en2)
#         ll = le1l[p1l[s1], :] .* le2l[p2l[s2], :]
#         @tensor LL[x, y] := LE[x, z, y] * ll[z]
#         BB = @view B[:, pd[s1], :]
#         RR = @view RE[:, pr[s2], :]
#         XX = reshape(LL' * BB * RR', size(LL, 2), 1, size(RR, 1))
#         lu = reshape(le1u[p1u[s1], :] .* le2u[p2u[s2], :], 1, :, 1)
#         L[:, :, :] += M.loc_exp[s2, s1] .* (XX .* lu)
#     end
#     L ./ maximum(abs.(L))
# end


"""
$(TYPEDSIGNATURES)
"""
function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}

    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs
    @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lb))

    A = zeros(size(LE, 3), maximum(p_rt), maximum(p_lt), size(RE, 1))
    for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
        le = @inbounds @view LE[:, l, :]
        b = @inbounds @view B4[:, p_lb[l], p_rb[r], :]
        re = @inbounds @view RE[:, r, :]
        @inbounds A[:,  p_rt[r], p_lt[l], :] += h[p_l[l], p_r[r]] .* (le' * b * re')
    end
    @cast AA[l, (ũ, u), r] := A[l, ũ, u, r]
    AA
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

"""
$(TYPEDSIGNATURES)
"""
function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: SparseSiteTensor}
    A = zeros(size(LE, 3), maximum(M.projs[4]), size(RE, 1))

    for (σ, lexp) ∈ enumerate(M.loc_exp)
        le = @inbounds @view LE[:, M.projs[1][σ], :]
        b = @inbounds @view B[:, M.projs[2][σ], :]
        re = @inbounds @view RE[:, M.projs[3][σ], :]
        @inbounds A[:, M.projs[4][σ], :] += lexp .* (le' * b * re')
    end
    A
end


"""
$(TYPEDSIGNATURES)
"""
function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: SparsePegasusSquareTensor}
    pl, pu, pr, pd = M.projs
    le1l, le2l, le1u, le2u = M.bnd_exp
    p1l, p2l, p1u, p2u = M.bnd_projs
    en1, en2 = M.loc_en
    L = zeros(size(LE, 3), maximum(pd), size(RE, 1))
    for s1 ∈ 1:length(en1), s2 ∈ 1:length(en2)
        ll = le1l[p1l[s1], :] .* le2l[p2l[s2], :]
        lu = le1u[p1u[s1], :] .* le2u[p2u[s2], :]
        @tensor LL[x, y] := LE[x, z, y] * ll[z]
        @tensor BB[x, y] := B[x, z, y] * lu[z]
        RR = @view RE[:, pr[s2], :]
        L[:, pd[s1], :] += M.loc_exp[s2, s1] .* (LL' * BB * RR')
    end
    L ./ maximum(abs.(L))
    # project_ket_on_bra(LE, B, M.M, RE, Val(:c))
end


"""
$(TYPEDSIGNATURES)
"""
function project_ket_on_bra(
    LE::S, B::S, M::T, RE::S, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: SparseVirtualTensor}
    h = M.con
    p_lb, p_l, p_lt, p_rb, p_r, p_rt = M.projs
    @cast B4[x, k, l, y] := B[x, (k, l), y] (k ∈ 1:maximum(p_lb))

    A = zeros(size(LE, 3), maximum(p_lb), maximum(p_rb), size(RE, 1))
    for l ∈ 1:length(p_l), r ∈ 1:length(p_r)
        le = @inbounds @view LE[:, l, :]
        b = @inbounds @view B4[:, p_rt[r], p_lt[l], :]
        re = @inbounds @view RE[:, r, :]
        @inbounds  A[:, p_lb[l], p_rb[r], :] += h[p_l[l], p_r[r]] .* (le' * b * re')
    end
    @cast AA[l, (ũ, u), r] := A[l, ũ, u, r]
    AA
end

"""
$(TYPEDSIGNATURES)
"""
function project_ket_on_bra(
    LE::S, B₀::S, M::T, RE::S, ::Val{:n}
) where {S <: AbstractArray{Float64, 3}, T <: AbstractDict}
    C = sort(collect(M), by = x -> x[1])
    TT = B₀
    for (_, v) ∈ reverse(C) TT = project_ket_on_bra(LE, TT, v, RE, Val(:n)) end
    TT
end

"""
$(TYPEDSIGNATURES)
"""
function project_ket_on_bra(
    LE::S, B₀::S, M::T, RE::S, ::Val{:c}
) where {S <: AbstractArray{Float64, 3}, T <: AbstractDict}
    C = sort(collect(M), by = x -> x[1])
    TT = B₀
    for (_, v) ∈ C TT = project_ket_on_bra(LE, TT, v, RE, Val(:c)) end
    TT
end

"""
$(TYPEDSIGNATURES)
"""
function measure_env(env::Environment, site::Site, trans::Symbol=:n)
    L = update_env_left(
        env.env[(site, :left)], env.bra[site], env.mpo[site], env.ket[site], trans
    )
    R = env.env[(site, :right)]
    @tensor L[t, c, b] * R[b, c, t]
end

"""
$(TYPEDSIGNATURES)
"""
function truncate!(ψ::QMps, s::Symbol, Dcut::Int=typemax(Int), tolS::Real=1E-16, args...)
    @assert s ∈ (:left, :right)
    if s == :right
        _right_sweep!(ψ, args...)
        _left_sweep!(ψ, Dcut, tolS, args...)
    else
        _left_sweep!(ψ, args...)
        _right_sweep!(ψ, Dcut, tolS, args...)
    end
end

"""
$(TYPEDSIGNATURES)
"""
canonise!(ψ::QMps, s::Symbol) = canonise!(ψ, Val(s))

"""
$(TYPEDSIGNATURES)
"""
canonise!(ψ::QMps, ::Val{:right}) = _left_sweep!(ψ, typemax(Int))

"""
$(TYPEDSIGNATURES)
"""
canonise!(ψ::QMps, ::Val{:left}) = _right_sweep!(ψ, typemax(Int))

"""
$(TYPEDSIGNATURES)
"""
function variational_sweep!(
    bra::QMps,
    mpo::QMpo,
    ket::QMps,
    ::Val{:left},
    trans::Symbol=:n,
    args...
)
    env = Environment(bra, mpo, ket, trans)
    _right_sweep_var!(env, trans, args...)
end


"""
$(TYPEDSIGNATURES)
"""
function variational_sweep!(
    bra::QMps,
    mpo::QMpo,
    ket::QMps,
    ::Val{:right},
    trans::Symbol=:n,
    args...
)
    env = Environment(bra, mpo, ket, trans)
    _left_sweep_var!(env, trans, args...)
end

function canonise_truncate!(ψ::QMps, s::Symbol, Dcut::Int=typemax(Int), tolS::Real=1E-16, args...)
    @assert s ∈ (:left, :right)
    if s == :right
        _left_sweep!(ψ, Dcut, tolS, args...)
    else
        _right_sweep!(ψ, Dcut, tolS, args...)
    end
end

"""
$(TYPEDSIGNATURES)
"""
function _right_sweep!(ψ::QMps, Dcut::Int=typemax(Int), tolS::Real=1E-16, args...)
    R = ones(eltype(ψ[1]), 1, 1)
    for i ∈ ψ.sites
        A = ψ[i]
        @matmul M̃[(x, σ), y] := sum(α) R[x, α] * A[α, σ, y]
        Q, R = qr_fact(M̃, Dcut, tolS, args...)
        R ./= maximum(abs.(R))
        @cast A[x, σ, y] := Q[(x, σ), y] (σ ∈ 1:size(A, 2))
        ψ[i] = A
    end
end

"""
$(TYPEDSIGNATURES)
"""
function _left_sweep!(ψ::QMps, Dcut::Int=typemax(Int), tolS::Real=1E-16, args...)
    R = ones(eltype(ψ[1]), 1, 1)
    for i ∈ reverse(ψ.sites)
        B = ψ[i]
        @matmul M̃[x, (σ, y)] := sum(α) B[x, σ, α] * R[α, y]
        R, Q = rq_fact(M̃, Dcut, tolS, args...)
        R ./= maximum(abs.(R))
        @cast B[x, σ, y] := Q[x, (σ, y)] (σ ∈ 1:size(B, 2))
        ψ[i] = B
    end
end

"""
$(TYPEDSIGNATURES)
"""
function _gauges_right_sweep!!!(ψ_top::QMps, ψ_bot::QMps, all_gauges::Dict)
    RT, RB = ones(1, 1), ones(1, 1)
    for i ∈ ψ_top.sites
        T, B = ψ_top[i], ψ_bot[i]

        @tensor T[a, b, c] := RT[a, s] * T[s, b, c]
        @tensor B[a, b, c] := RB[a, s] * B[s, b, c]
        @tensor ρ_t[r, s] := T[i, r, j] * conj(T)[i, s, j]
        @tensor ρ_b[r, s] := B[i, r, j] * conj(B)[i, s, j]

        gauge = (diag(ρ_b) ./ diag(ρ_t)) .^ (1 / 4) # optimize
        gauge_inv = 1.0 ./ gauge
        all_gauges[i] .*= gauge # update

        AT = T .* reshape(gauge, (1, :, 1))
        AB = B .* reshape(gauge_inv, (1, :, 1))

        @cast ATR[(x, σ), y] := AT[x, σ, y]
        QT, RT = qr_fact(ATR)
        RT ./= maximum(abs.(RT))
        @cast AT[x, σ, y] := QT[(x, σ), y] (σ ∈ 1:size(AT, 2))
        ψ_top[i] = AT

        @cast ABR[(x, σ), y] := AB[x, σ, y]
        QB, RB = qr_fact(ABR)
        RB ./= maximum(abs.(RB))
        @cast AB[x, σ, y] := QB[(x, σ), y] (σ ∈ 1:size(AB, 2))
        ψ_bot[i] = AB
    end
end

"""
$(TYPEDSIGNATURES)
"""
function _gauges_left_sweep!!!(ψ_top::QMps, ψ_bot::QMps, all_gauges::Dict)
    RT, RB = ones(1, 1), ones(1, 1)

    for i ∈ reverse(ψ_top.sites)
        T, B = ψ_top[i], ψ_bot[i]

        @tensor T[a, b, c] := T[a, b, s] * RT[s, c]
        @tensor B[a, b, c] := B[a, b, s] * RB[s, c]
        @tensor ρ_t[r, s] := T[i, r, j] * conj(T)[i, s, j]
        @tensor ρ_b[r, s] := B[i, r, j] * conj(B)[i, s, j]

        gauge = (diag(ρ_b) ./ diag(ρ_t)) .^ (1 / 4) # optimize
        gauge_inv = 1.0 ./ gauge
        all_gauges[i] .*= gauge # update

        AT = T .* reshape(gauge, (1, :, 1))
        AB = B .* reshape(gauge_inv, (1, :, 1))

        @cast ATR[x, (σ, y)] := AT[x, σ, y]
        RT, QT = rq_fact(ATR)
        RT ./= maximum(abs.(RT))
        @cast AT[x, σ, y] := QT[x, (σ, y)] (σ ∈ 1:size(AT, 2))
        ψ_top[i] = AT

        @cast ABR[x, (σ, y)] := AB[x, σ, y]
        RB, QB = rq_fact(ABR)
        RB ./= maximum(abs.(RB))
        @cast AB[x, σ, y] := QB[x, (σ, y)] (σ ∈ 1:size(AB, 2))
        ψ_bot[i] = AB
    end
end

"""
$(TYPEDSIGNATURES)
"""
function optimize_gauges_for_overlaps!!(
    ψ_top::QMps,
    ψ_bot::QMps,
    tol::Real=1E-8,
    max_sweeps::Int=4
)
    canonise!(ψ_top, :right)
    canonise!(ψ_bot, :right)

    overlap_old = dot(ψ_top, ψ_bot)
    all_gauges = Dict(i => ones(size(ψ_top[i], 2)) for i ∈ ψ_top.sites)
    for _ ∈ 1:max_sweeps
        _gauges_right_sweep!!!(ψ_top, ψ_bot, all_gauges)
        _gauges_left_sweep!!!(ψ_top, ψ_bot, all_gauges)

        overlap_new = dot(ψ_top, ψ_bot)
        Δ = overlap_new / overlap_old
        overlap_old = overlap_new
        if abs(Δ - 1.0) < tol break end
    end
    all_gauges
end
