using CUDA


function test_kernel(A, b)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    x_stride = gridDim().x * blockDim().x

    for k ∈ idx:x_stride:size(A, 2)
        @inbounds A[:, k] += b[k]
    end
end

n, k = 32, 256

A = CUDA.rand(n, k)
b = CUDA.rand(k)

th = 256
bl = ceil(Int, k / th)

@time begin
    CUDA.@sync begin
        @cuda threads=th blocks=bl test_kernel(A, b)
    end
end
