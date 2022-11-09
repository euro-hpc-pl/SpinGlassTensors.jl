using NNlib, NNlibCUDA
using CUDA
using LoopVectorization
using BatchedRoutines

function batched_gemm!(C, A, B)
    @turbo for m ∈ axes(A, 1), n ∈ axes(B, 2), i ∈ axes(B, 3)
        Cmni = zero(eltype(C))
        for k ∈ axes(A, 2)
            Cmni += A[m, k, i] * B[k, n, i]
        end
        C[m, n, i] = Cmni
    end
end

σ = 256
η = 256

a = rand(σ, σ, η)
b = rand(σ, σ, η)

@time d = a ⊠ b

@time begin
    c = zeros(σ, σ, η)
    #Threads.@threads
    for σ ∈ 1:size(a, 3)
        c[:, :, σ] = a[:, :, σ] * b[:, :, σ]
    end
end

e = zeros(σ, σ, η)
@time batched_gemm!(e, a, b)

f = zeros(σ, σ, η)
α = 1.0
β = 0.0
@time BatchedRoutines.batched_gemm!('N', 'N', α, a, b, β, f)

# GPU
a_d = CUDA.CuArray(a)
b_d = CUDA.CuArray(b)

@time d_d = batched_mul(a_d, b_d)

@assert c ≈ d ≈ e ≈ f
