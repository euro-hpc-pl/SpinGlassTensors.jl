#TODO add support for CuSparseMatrixCSR (cf. https://github.com/JuliaGPU/CUDA.jl/issues/1113)

@memoize Dict function aux_cusparse(::Type{R}, n::Int64) where R <: Real
    CuArray(1:n+1), CUDA.ones(R, n)
end

function SparseCSC(::Type{R}, p::CuArray{Int64, 1}) where R <: Real
    n = length(p)
    mp = maximum(p)
    cn, co = aux_cusparse(R, n)
    CuSparseMatrixCSC(cn, p, co, (mp, n))
end

function SparseCSC(::Type{R}, p::Vector{Int64}) where R <: Real
    n = length(p)
    mp = maximum(p)
    cn = collect(1:n)
    co = ones(R, n)
    sparse(p, cn, co, mp, n)
end

function SparseCSC(::Type{T}, p1::R, p2::R, p3::R) where {T <: Real, R <: Union{CuArray{Int64, 1}, Vector{Int64}}}
    @assert length(p1) == length(p2) == length(p3)
    s1, s2 = maximum(p1), maximum(p2)
    p = p1 .+ s1 * (p2 .- 1) .+ s1 * s2 * (p3 .- 1)
    SparseCSC(T, p)
end



@memoize Dict function SparseCSCslice(::Type{R}, lp::PoolOfProjectors, p::Int, from::Int, to::Int, device::Symbol) where R <: Real
    pout = get_projector!(lp, p, device)
    poutp = @view pout[from:to]
    rf = minimum(poutp)
    rt = maximum(poutp)
    ipr = SparseCSC(R, poutp .- (rf - 1))
    (ipr, rf, rt)
end