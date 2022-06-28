using MKL_jll, OpenBLAS_jll, Libdl, LinearAlgebra

const libMKL = Libdl.dlopen(MKL_jll.libmkl_rt)
const DGEMM = Libdl.dlsym(libMKL, :dgemm)

function mkl_dgemm()
    M, N, K = 32, 32, 32
    a_array = rand(M, K)
    b_array = rand(K, N)
    c_array = zeros(M, N)

    @time a_array * b_array

    ccall(
        DGEMM, Cvoid,
        (
            Cchar, Cchar,
            Cint, Cint, Cint,
            Float64,
            Ptr{Float64}, Cint,
            Ptr{Float64}, Cint,
            Float64,
            Ptr{Float64}, Cint,
        ),
        'N', 'N',
        M, N, K,
        1.0,
        a_array, M,
        b_array, K,
        0.0,
        c_array, M
    )
end

mkl_dgemm()
