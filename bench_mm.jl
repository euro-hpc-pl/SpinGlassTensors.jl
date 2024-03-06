using TensorCast, TensorOperations
function time_mm()
    M = rand(100, 100, 100)
    L = rand(100, 100)
    R = rand(100, 100)
    @time begin
        @matmul M1[x, σ, α] := sum(β) L[x, β] * M[β, σ, α] 
        @matmul MM[x, σ, y] := sum(α) M1[x, σ, α] * R[α, y]
    end
end

function time_tensor()
    M = rand(100, 100, 100)
    L = rand(100, 100)
    R = rand(100, 100)

    @time begin
        @tensor M̃[x, σ, y] := L[x, β] * M[β, σ, α] * R[α, y] order = (α, β)
        # @cast B[(x, σ), y] |= M̃[x, σ, y]
    end
end

println("matmul")
time_mm()
time_mm()

println("\n tensor")
time_tensor()
time_tensor()
