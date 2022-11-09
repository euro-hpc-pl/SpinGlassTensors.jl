using MKL_jll, OpenBLAS_jll, Libdl, LinearAlgebra

const libMKL = Libdl.dlopen(MKL_jll.libmkl_rt)
const DGEMM_BATCH = Libdl.dlsym(libMKL, :dgemm_batch)

function dgemm_batch()
    group_count = 2
    group_size = fill(1, group_count)

    M, N, K = Int32(32), Int32(32), Int32(32)

    a_array = fill(rand(M, K), group_count)
    b_array = fill(rand(K, N), group_count)
    c_array = fill(Matrix{Float64}(undef, M, N), group_count)

    lda_array = Int32.(stride.(a_array, Ref(2)))
    ldb_array = Int32.(stride.(b_array, Ref(2)))
    ldc_array = Int32.(stride.(c_array, Ref(2)))

    m_array = fill(M, group_count)
    n_array = fill(N, group_count)
    k_array = fill(K, group_count)

    transa_array = fill(UInt8('N'), group_count)
    transb_array = fill(UInt8('N'), group_count)

    alpha_array = fill(one(Float64), group_count)
    beta_array = fill(zero(Float64), group_count)

    @time a_array .* b_array
    @time d_array = a_array .* b_array

    @time ccall(
        DGEMM_BATCH, Cvoid,
        (
            Ptr{Cchar}, Ptr{Cchar}, #transa_array, transb_array
            Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, #m_array, n_array, k_array
            Ptr{Cdouble}, #alpha_array
            Ptr{Cdouble}, Ptr{Cint}, #a_array, lda_array
            Ptr{Cdouble}, Ptr{Cint}, #b_array, ldb_array
            Ptr{Cdouble}, #beta_array
            Ptr{Cdouble}, Ptr{Cint}, #c_array, ldc_array
            Ref{Cint}, Ptr{Cint}, #group_count, group_size
        ),
        transa_array, transb_array,
        m_array, n_array, k_array,
        alpha_array,
        a_array, lda_array,
        b_array, ldb_array,
        beta_array,
        c_array, ldc_array,
        group_count, group_size
    )
end

dgemm_batch()
