using NNlib

σ = 32
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

@assert c ≈ d
