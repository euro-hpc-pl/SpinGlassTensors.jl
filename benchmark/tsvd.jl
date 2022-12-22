using TSVD
using TensorOperations
using TensorCast
using LinearAlgebra


# C = A * B
struct MyTensor{T <: Real}
    A::Array{T, 2}
    B::Array{T, 2}
end

Base.eltype(ten::MyTensor{T}) where T = T
Base.size(ten::MyTensor{T}) where T = (size(ten.A, 1), size(ten.B, 2))
Base.size(ten::MyTensor{T}, n::Int) where T = size(ten)[n]
Base.adjoint(ten::MyTensor{T}) where T = MyTensor{T}(adjoint(ten.B), adjoint(ten.A))
Base.:(*)(ten::MyTensor{T}, v::Vector{T}) where T = (ten.A * (ten.B * v))

n = 100
cut = 10
T = Float64

ten = MyTensor(rand(T, n, n), rand(T, n, n))
@time U, Σ, V = tsvd(ten, cut)

# basic example
#=
A = rand(T, n, n)

@time U, Σ, V = tsvd(A, cut)
=#

nothing
