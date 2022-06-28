using MKL_jll, Libdl

const libMKL = Libdl.dlopen(MKL_jll.libmkl_rt)
const DGEMM = Libdl.dlsym(libMKL, :dgemm)

function mkl_dgemm()
    M, N, K = 32, 32, 32
    a_array = rand(M, K)
    b_array = rand(K, N)
    c_array = zeros(M, N)

    @time d = a_array * b_array

    @time ccall(
        DGEMM, Cvoid,
        (
            Ref{Cchar}, Ref{Cchar},
            Ref{Cint}, Ref{Cint}, Ref{Cint},
            Ref{Cdouble},
            Ptr{Cdouble}, Ref{Cint},
            Ptr{Cdouble}, Ref{Cint},
            Ref{Cdouble},
            Ptr{Cdouble}, Ref{Cint},
        ),
        'N', 'N',
        M, N, K,
        1.0,
        a_array, M,
        b_array, K,
        0.0,
        c_array, M
    )

    @assert c_array ≈ d
end

mkl_dgemm()
