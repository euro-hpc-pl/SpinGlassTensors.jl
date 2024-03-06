using Memoization
using LinearAlgebra
using CUDA
using BenchmarkTools

# Functions from constactions_cuda/sparse.jl which are not exported

@memoize Dict function aux_cusparse(::Type{R}, n::Int64) where {R<:Real}
    println("entering aux_cusparse function")
    CuArray(1:n+1), CUDA.ones(R, n)
end

@memoize Dict function CUDA.CUSPARSE.CuSparseMatrixCSC(
    ::Type{R},
    p::Vector{Int},
) where {R<:Real}
    println("entering cusparse")
    n = length(p)
    cn, co = aux_cusparse(R, n)
    CUDA.CUSPARSE.CuSparseMatrixCSC(cn, CuArray(p), co, (maximum(p), n))
end


function CuSparseMatrixCSC_no_memo(::Type{R}, p::Vector{Int}) where {R<:Real}
    println("entering no memo")
    n = length(p)
    cn, co = aux_cusparse(R, n)
    CUDA.CUSPARSE.CuSparseMatrixCSC(cn, CuArray(p), co, (maximum(p), n))
end

# test of their memoization

p = sort(rand(1:5000, 10000000))
p2 = sort(rand(1:5000, 10000000))
@time A = CuSparseMatrixCSC_no_memo(Float64, p)
@time B = CuSparseMatrixCSC_no_memo(Float64, p)

@time C = CUDA.CUSPARSE.CuSparseMatrixCSC(Float64, p) # compilation time?

@time D = CUDA.CUSPARSE.CuSparseMatrixCSC(Float64, p)
@time E = CUDA.CUSPARSE.CuSparseMatrixCSC(Float64, p2)
@time F = CUDA.CUSPARSE.CuSparseMatrixCSC(Float64, p2)
CUDA.memory_status()
Memoization.empty_all_caches!()
CUDA.memory_status()
# clearing memoization caches doeas not free GPU memory
