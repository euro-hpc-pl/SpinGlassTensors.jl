using CUDA

n, m, k, p = 32, 32, 1024, 1024

A = rand(n, m, k);
A_d = CUDA.CuArray(A)

println("CPU:")

@time x = A .* ones(1, 1, 1, p)
@time y = repeat(A, outer=(1, 1, 1, p))

@assert x ≈ y

println("GPU:")

@time x = A_d .* CUDA.ones(Float64, 1, 1, 1, p)
@time y = repeat(A_d, outer=(1, 1, 1, p))

nothing
