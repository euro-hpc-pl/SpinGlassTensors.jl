using LinearAlgebra, MKL
using TensorOperations
using TensorCast
using TSVD
using LowRankApprox
using RandomizedLinAlg
using FameSVD


# C = A * B
struct MyTensor{T<:Number}
    A::Array{T,2}
    B::Array{T,2}
end

Base.Array(ten::MyTensor) = ten.A * ten.B

# this is for tsvd to work
Base.eltype(ten::MyTensor{T}) where {T} = T
Base.size(ten::MyTensor) = (size(ten.A, 1), size(ten.B, 2))
Base.size(ten::MyTensor, n::Int) = size(ten)[n]
Base.adjoint(ten::MyTensor{T}) where {T} = MyTensor{T}(adjoint(ten.B), adjoint(ten.A))
Base.:(*)(ten::MyTensor{T}, v::Vector{T}) where {T} = (ten.A * (ten.B * v))

# this is for psvd to work
LinearAlgebra.ishermitian(ten::MyTensor) = ishermitian(ten.A) && ishermitian(ten.B)
LinearAlgebra.mul!(y, ten::MyTensor, x) = mul!(y, ten.B, ten.A * x)

n = 2^12
cut = 2^6
T = Float64

ten = MyTensor(rand(T, n, n), rand(T, n, n))

println("MyTensor:")
println("tsvd:")
@time U, Σ, V = tsvd(ten, cut)

println("psvd:")
@time U, Σ, V = psvd(ten, rank = cut)

println("Array:")
println("tsvd:")
@time begin
    C = Array(ten)
    U, Σ, V = tsvd(C, cut)
end

println("svd:")
@time begin
    C = Array(ten)
    U, Σ, V = svd(C)
end

println("psvd:")
@time begin
    C = Array(ten)
    U, Σ, V = psvd(C, rank = cut)
end

println("rsvd:")
@time begin
    C = Array(ten)
    U, Σ, V = rsvd(C, cut, 0)
end

println("fsvd:")
@time begin
    C = Array(ten)
    U, Σ, V = fsvd(C)
end

nothing
