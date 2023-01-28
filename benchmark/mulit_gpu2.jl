using CUDA


T = Float64
n = 100
gpus = Int(length(devices()))

a = rand(T, n, n, gpus)
a_d = cu([1,2,3], unified=true)

a_d
