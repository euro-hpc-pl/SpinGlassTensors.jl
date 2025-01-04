using cuTENSOR
using CUDA, CUDA.CUSPARSE
using LinearAlgebra, MKL

T = Float64
onGPU = true

A1 = rand(T, 1, 2)
A2 = rand(T, 1, 2)

B = rand(T, 1, 1)
C = rand(T, 2, 2, 2, 1)

if onGPU
    A1 = cu(A1)
    A2 = cu(A2)
    B = cu(B)
    C = cu(C)
end

mul!(A1, B, (@view C[1, 2, :, :])')
mul!(A2, B, C[1, 2, :, :]')

println(A1)
println(A2)
