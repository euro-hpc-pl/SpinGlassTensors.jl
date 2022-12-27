using LinearAlgebra, MKL
using TensorOperations
using TensorCast
using TSVD
using LowRankApprox
using RandomizedLinAlg
using FameSVD


# C = A * B
struct MyTensor{T <: Number}
    A::Array{T, 2}
    B::Array{T, 2}
end

struct AMyTensor{T <: Number}
    A::Array{T, 2}
    B::Array{T, 2}
end


Base.Array(ten::MyTensor) = kron(ten.A, ten.B)

# this is for tsvd to work
Base.eltype(ten::MyTensor{T}) where T = T
Base.size(ten::MyTensor) = (size(ten.A, 1) * size(ten.B, 1), size(ten.A, 2) * size(ten.B, 2))
Base.size(ten::MyTensor, n::Int) = size(ten)[n]
# Base.adjoint(ten::MyTensor{T}) where T = MyTensor{T}(adjoint(ten.A), adjoint(ten.B))

# Base.:(*)(ten::MyTensor{T}, v::Vector{T}) where T = (kron(ten.A, ten.B) * v)

Base.adjoint(ten::MyTensor{T}) where T = AMyTensor{T}(ten.A, ten.B)


function Base.:(*)(ten::MyTensor{T}, v::Vector{T}) where T 
    println("M")
    vv = reshape(v, size(ten.A, 2), size(ten.B, 2))
    @tensor x[x1, y1] := ten.A[x1, x2] * ten.B[y1, y2] * vv[x2, y2]
    reshape(x, size(ten.A, 1) * size(ten.B, 1))
end

function Base.:(*)(ten::AMyTensor{T}, v::Vector{T}) where T 
    println("A")
    vv = reshape(v, size(ten.A, 1), size(ten.B, 1))
    @tensor x[x1, y1] := ten.A[x2, x1] * ten.B[y2, y1] * vv[x2, y2]
    reshape(x, size(ten.A, 2) * size(ten.B, 2))
end


# this is for psvd to work
LinearAlgebra.ishermitian(ten::MyTensor) = false


function LinearAlgebra.mul!(y, ten::MyTensor, v) 
    println("K")
    vv = reshape(v, size(ten.A, 2), size(ten.B, 2), :)
    @tensor x[x1, y1, z1] := ten.A[x1, x2] * ten.B[y1, y2] * vv[x2, y2, z1]
    y[:, :] = reshape(x, size(ten.A, 1) * size(ten.B, 1), :)
end

function LinearAlgebra.mul!(y, ten::AMyTensor, v) 
    println("L")
    vv = reshape(v, size(ten.A, 1), size(ten.B, 1), :)
    @tensor x[x1, y1, z1] := ten.A[x2, x1] * ten.B[y2, y1] * vv[x2, y2, z1]
    y[:, :] = reshape(x, size(ten.A, 2) * size(ten.B, 2), :)
end


n = 2 ^ 2
cut = 2 ^ 1
T = Float64

ten = MyTensor(rand(T, n+1, n), rand(T, n+2, n-1))

println("tsvd:")
@time U, Σ1, V = tsvd(ten, cut)

println("psvd:")
@time U, Σ2, V = psvd(ten, rank=cut)

println("svd:")
@time begin
    C = Array(ten)
    U, Σ3, V = svd(C)
end

# println("psvd:")
# @time begin
#     C = Array(ten)
#     U, Σ4, V = psvd(C, rank=cut)
# end

println(Σ1)
println(Σ2)
println(Σ3[1:cut])
# println(Σ4)