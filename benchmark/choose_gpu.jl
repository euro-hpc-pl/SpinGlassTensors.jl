using CUDA
device_list = [device for device in devices()]
println(device_list)

device!(0)
A = CUDA.rand(Float64, (100, 100))
println(device(A)) # prints on wchich device array A is located

if length(device_list) >= 2
    device!(1)
    B = CUDA.rand(Float64, (100, 100))
    println(device(B)) # prints on wchich device array B is located
end

if length(device_list) >= 3
    device!(2)
    C = CUDA.rand(Float64, (100, 100))
    println(device(C)) # prints on wchich device array C is located
end

# We cannot multiply arrays on different devices (or take any actions between them). Every array is tied to its device
# Can be made pararel with Distributed package