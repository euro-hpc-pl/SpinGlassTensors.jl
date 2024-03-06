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
S = exp.(collect(0:N-1) * log(4 / 5));

mat = U * Diagonal(S) * V';
U, S, V = svd(mat);

U, S, V = U[:, 1:cut], S[1:cut], V[:, 1:cut]
mat1 = U * Diagonal(S) * V'
println(S[1:cut])
println(norm(mat - mat1))

Up, Sp, Vp = psvd(mat, rank = 2 * cut)

mat2 = Up[:, 1:cut] * Diagonal(Sp[1:cut]) * Vp[:, 1:cut]'

println(Sp[1:cut])
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
Uf, Sf, Vf = Uf[:, 1:cut], Sf[1:cut], Vf[:, 1:cut]
mat3 = Uf * Diagonal(Sf) * Vf' * Vp'
println(Sf - S[1:cut])
println(norm(mat - mat3))

nothing


iter = 5
Up, Sp, Vp = [], [], []
for i = 1:iter
    Utemp, Stemp, Vtemp = psvd(mat, rank = 2 * cut)
    push!(Up, Utemp)
    push!(Sp, Stemp)
    push!(Vp, Vtemp)
end

Ups = hcat(Up...)
Vps = hcat(Vp...)
Sps = vcat(Sp...) / iter
println(size(Ups), " ", size(Vps), " ", size(Sps))
println(size(Up[1]), " ", size(Vp[1]), " ", size(Sp[1]))

Uq, Ur = qr(Ups)
Vq, Vr = qr(Vps)

Ut, St, Vt = svd(Ur * Diagonal(Sps) * Vr')

U2 = Uq * Ut[:, 1:cut]
V2 = Vq * Vt[:, 1:cut]
S2 = St[1:cut]
println(St)
println(S2)

mat4 = U2 * Diagonal(S2) * V2'


println(norm(mat1 - mat2))
println(norm(mat1 - mat3))
println(norm(mat1 - mat4))
