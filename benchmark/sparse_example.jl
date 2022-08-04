
using CUDA
using CUDA.CUSPARSE
using SparseArrays
using TensorOperations
using TensorCast

T = Float64
n = 1
N = 3

a = rand(T, n, n); a[a .< T(9//10)] .= T(0)
a_d = CuSparseMatrixCSC(sparse(a))



pr = sort(rand(1:N, N^2))

b = CUDA.rand(T, N^2, n, n)
c = deepcopy(b)

#=
bb = CUDA.rand(T, n, n, n)

@time begin
    @tensor c[x, y, z] := b[x, y, s] * a_d[s, z]
end

@time begin
    @matmul c[x, y, z] := sum(s) bb[x, y, s] * a_d[s, z]
end

@time  a_d * b 
nothing

@time begin
    a = spzeros(maximum(pr), size(pr, 1))

    for (index, i) in enumerate(pr)
            a[i, index] = 1
        end
        a = CuSparseMatrixCSC(a)

        a*b 
    nothing
end
=#
@time begin
    @assert all(b .== c)

    #CUDA.memory_status()
    csrRowPtr = CuArray(collect(1:length(pr) + 1))
    #CUDA.memory_status()
    csrColInd = CuArray(pr)
    #CUDA.memory_status()
    csrNzVal = CUDA.ones(Float64, length(pr))
    #CUDA.memory_status()
    a = CUSPARSE.CuSparseMatrixCSR(csrRowPtr, csrColInd, csrNzVal, (maximum(pr), length(pr)))
    a2 = CUSPARSE.CuSparseMatrixCSR(csrRowPtr, csrColInd, csrNzVal, (length(pr), maximum(pr)))
    
    #CUDA.memory_status()
    
    #@cast c[x, (y, z)] := c[x, y, z]
    c = reshape(c, (N^2, :))
    d = a * c #(N, N^2) * (N^2, n^2)  =(N, n^2) 
    #d = reshape(d, (N, n, n))
    
    b = permutedims(b, (3, 2, 1)) #(n, n, N^2)
    #CUDA.memory_status()
    #@cast b[(x, y), z] := b[x, y, z] #(n^2, N^2)
    b = reshape(b, (:, N^2))
   
    e = b * a2 #(n^2, N^2) * (N^2, N) = (n^2, N)
    e = permutedims(e, (2,1))
    #@assert isequal(d,e)
    #e = reshape(e, (N, n, n))
    
    println(d)
    println(e)

end
