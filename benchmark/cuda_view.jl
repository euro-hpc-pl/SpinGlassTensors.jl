using CUDA

#println(CUDA.versioninfo())

T = Float64
n = 2 ^ 14
k = 2 ^ 6

#K = reverse(collect(1:k))
K = collect(1:k)
K_d = CuArray(K)

a = CUDA.rand(T, n, n)

#@time b1 = a[:, K]
#@time b2 = view(a, :, K)
#@time b3 = view(a, :, K_d)

c = CUDA.zeros(T, n, k)
for j ∈ 1:k
    c[:, j] = @view a[:, j]
end

c[1, 1] = -1.0
println(a[1,1])

#println(typeof(b1))


c = a * c;
nothing
