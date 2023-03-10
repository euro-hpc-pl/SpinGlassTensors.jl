using LinearAlgebra, MKL
using TensorOperations
using TensorCast
using TSVD
using LowRankApprox
using RandomizedLinAlg
using FameSVD

N = 100
cut = 8

mat = rand(100, 100);
U, S, V = svd(mat);
S = exp.(collect(0:N-1) * log(4/5));

mat = U * Diagonal(S) * V';
U, S, V = svd(mat);

U, S, V  = U[:, 1:cut], S[1:cut], V[:, 1:cut] 
mat1 = U * Diagonal(S) * V'
println(S[1:cut])
println(norm(mat - mat1))

Up, Sp, Vp = psvd(mat, rank=2*cut)

mat2 = Up[:, 1:cut] * Diagonal(Sp[1:cut]) * Vp[:, 1:cut]'
println(Sp[1:cut] - S[1:cut])
println(norm(mat - mat2))

   # Vp = V

    C = mat * Vp
    println(size(C))
    Ut, _ = qr(C)
    Ut = Ut[:, 1:cut]
    println(size(Ut))
    C = Ut' * mat
    Vp, _ = qr(C')
    Vp = Vp[:, 1:cut]



    C = mat * Vp
    Uf, Sf, Vf = svd(C);
    Uf, Sf, Vf  = Uf[:, 1:cut], Sf[1:cut], Vf[:, 1:cut] 
    mat3 = Uf * Diagonal(Sf) * Vf' * Vp'

    println(Sf - S[1:cut])
    println(norm(mat - mat3))

   nothing
