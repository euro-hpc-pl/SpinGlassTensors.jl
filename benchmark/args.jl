using LinearAlgebra

function my_svd(A; kwargs...)
    svd(A; kwargs...)
end


T = Float64
n = 2
A = rand(T, 2, 2)

my_svd(A, full = true)
