
# transfer.jl: This file provides rules of how to transfer tensors to GPU. Note, NOT all of
#              tensor's coponents are moved from CPU to GPU and most tensors are generated
#              on CPU due to the size of factor graph.
export
    device,
    move_to_CUDA!

move_to_CUDA!(ten::Array{T, N}) where {T, N} = CuArray(ten)
move_to_CUDA!(ten::CuArray{T, N}) where {T, N} = ten
move_to_CUDA!(ten::Diagonal) = Diagonal(move_to_CUDA!(diag(ten)))

function move_to_CUDA!(ten::CentralTensor)
    ten.e11 = move_to_CUDA!(ten.e11)
    ten.e12 = move_to_CUDA!(ten.e12)
    ten.e21 = move_to_CUDA!(ten.e21)
    ten.e22 = move_to_CUDA!(ten.e22)
    ten
end

function move_to_CUDA!(ten::DiagonalTensor)
    ten.e1 = move_to_CUDA!(ten.e1)
    ten.e2 = move_to_CUDA!(ten.e2)
    ten
end

function move_to_CUDA!(ten::VirtualTensor)
    ten.con = move_to_CUDA!(ten.con)
    ten
end

function move_to_CUDA!(ten::SiteTensor)
    ten.loc_exp = move_to_CUDA!(ten.loc_exp)
    ten
end

move_to_CUDA!(ten::Nothing) = ten

function move_to_CUDA!(ten::MpoTensor)
    for i ∈ 1:length(ten.top) ten.top[i] = move_to_CUDA!(ten.top[i]) end
    for i ∈ 1:length(ten.bot) ten.bot[i] = move_to_CUDA!(ten.bot[i]) end
    ten.ctr = move_to_CUDA!(ten.ctr)
    ten
end

function move_to_CUDA!(ψ::Union{QMpo{T}, QMps{T}}) where T
    for k ∈ keys(ψ.tensors) move_to_CUDA!(ψ[k]) end
    ψ
end

device(ten::Nothing) = Set()
device(ψ::Union{QMpo{T}, QMps{T}}) where T = union(device.(values(ψ.tensors))...)
device(ten::MpoTensor) = union(device(ten.ctr), device.(ten.top)..., device.(ten.bot)...)
device(ten::DiagonalTensor) = union(device.((ten.e1, ten.e2))...)
device(ten::VirtualTensor) = device(ten.con)
device(ten::CentralTensor) = union(device.((ten.e11, ten.e12, ten.e21, ten.e22))...)
device(ten::SiteTensor) = device(ten.loc_exp)
device(ten::Array{T, N}) where {T, N} = Set((:CPU, ))
device(ten::CuArray{T, N}) where {T, N} = Set((:GPU, ))
device(ten::Diagonal) = device(diag(ten))
