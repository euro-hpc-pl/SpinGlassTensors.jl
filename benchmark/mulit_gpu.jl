using CUDA

function move_to_CUDA(a::Array{T,N}) where {T,N}
    buf_a = Mem.alloc(Mem.Unified, sizeof(a))
    d_a = unsafe_wrap(CuArray{T,N}, convert(CuPtr{T}, buf_a), size(a))
    finalizer(d_a) do _
        Mem.free(buf_a)
    end
    copyto!(d_a, a)
    d_a
end

T = Float64
n = 100
gpus = Int(length(devices()))

a = rand(T, n, n, gpus)
a_d = move_to_CUDA(a)

for (gpu, dev) ∈ enumerate(devices())
    device!(dev)
    @views a_d[:, :, gpu] .= 2 .* a_d[:, :, gpu]
end

a_d
