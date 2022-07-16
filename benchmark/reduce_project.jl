using CUDA
using TensorOperations

n, m, k = 32, 32, 4096

L = rand(n, m, k);
L_d = CUDA.CuArray(A)

@time begin
    ipr = CUDA.CuArray(diagm(ones(maximum(pr)))[pr, :])
    @tensor ret[x, y, r] := Lnew[x, y, z] * ipr[z, r]
end

@time add_project(L, pr, th=(16, 16))

nothing
