using CUDA


function test_kernel(A, v, C)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    x_stride = gridDim().x * blockDim().x

    for k ∈ idx:x_stride:size(C, 3)
        for i ∈ 1:size(C, 1), j ∈ 1:size(C, 2)
            @inbounds C[i, j, k] += A[i, j, k]
        end
    end
    nothing
end

n, m, k = 32, 32, 256

A = CUDA.rand(n, m, k)
v = CUDA.rand(k)
C = CUDA.zeros(Float64, n, m, k)

th = 256
bl = ceil(Int, k / th)

@time begin
    CUDA.@sync begin
        @cuda threads=th blocks=bl test_kernel(A, v, C)
    end
end
