using CUDA

T = Float64
n = 100
gpus = Int(length(devices()))

a = rand(T, n, n, gpus)
a_d = cu(a, unified=true)

a_d 
