export
    Tensor,
    SiteTensor,
    VirtualTensor,
    DiagonalTensor,
    CentralTensor,
    CentralOrDiagonal,
    mpo_transpose

abstract type AbstractSparseTensor{T, N} end

const Proj{N} = NTuple{N, Array{Int, 1}}
const CuArrayOrArray{T, N} = Union{AbstractArray{T, N}, CuArray{T, N}} #TODO clean this !!!

# Allow data to reside on CUDA ???

move_to_CUDA!(ten::Array{T, N}) where {T, N} = CuArray(ten)
move_to_CUDA!(ten::Diagonal) = Diagonal(CuArray(diag(ten)))

device(ten::Array{T, N}) where {T, N} = Set((:CPU,))
device(
    ten::Union{CuArray, Diagonal{T, CuArray{T, 1, CUDA.Mem.DeviceBuffer}}}
) where T = Set((:GPU,)) # this is ugly but works for now

device(ten::Diagonal) = device(diag(ten))

ArrayOrCuArray(L) = typeof(L) <: CuArray ? CuArray : Array # TODO do we need this?

mutable struct SiteTensor{T <: Real, N} <: AbstractSparseTensor{T, N}
    loc_exp # ::Vector{T}
    projs::Proj{4}  # == pl, pt, pr, pb
    dims::Dims{N}

    function SiteTensor(loc_exp, projs::Proj{4}; dims=maximum.(projs))
        T = eltype(loc_exp)
        new{T, 4}(loc_exp, projs, dims)
    end
end

device(ten::SiteTensor) = device(ten.loc_exp)


function move_to_CUDA!(ten::SiteTensor)
    ten.loc_exp = move_to_CUDA!(ten.loc_exp)
    ten
end

function mpo_transpose(ten::SiteTensor)
    perm = [1, 4, 3, 2]
    SiteTensor(ten.loc_exp, ten.projs[perm], dims=ten.dims[perm])
end

mutable struct CentralTensor{T <: Real, N} <: AbstractSparseTensor{T, N}
    e11 #::Matrix{T}
    e12 #::Matrix{T}
    e21 #::Matrix{T}
    e22 #::Matrix{T}
    dims::Dims{N}

    function CentralTensor(e11, e12, e21, e22)
        s11, s12, s21, s22 = size.((e11, e12, e21, e22))
        @assert s11[1] == s12[1] && s21[1] == s22[1] &&
                s11[2] == s21[2] && s12[2] == s22[2]
        dims = (s11[1] * s21[1], s11[2] * s12[2])
        T = promote_type(eltype.((e11, e12, e21, e22))...)
        new{T, 2}(e11, e12, e21, e22, dims)
    end
end

device(ten::CentralTensor) = union(device(ten.e11), device(ten.e12), device(ten.e21), device(ten.e22))

function move_to_CUDA!(ten::CentralTensor)
    ten.e11 = move_to_CUDA!(ten.e11)
    ten.e12 = move_to_CUDA!(ten.e12)
    ten.e21 = move_to_CUDA!(ten.e21)
    ten.e22 = move_to_CUDA!(ten.e22)
    ten
end


mpo_transpose(ten::CentralTensor) = CentralTensor(transpose.((ten.e11, ten.e21, ten.e12, ten.e22))...)

const MatOrCentral{T, N} = Union{Matrix{T}, CentralTensor{T, N}}

# TODO should this be removed?
function Base.Array(ten::CentralTensor)
    @cast V[(u1, u2), (d1, d2)] := ten.e11[u1, d1] * ten.e21[u2, d1] *
                                   ten.e12[u1, d2] * ten.e22[u2, d2]
    V ./ maximum(V)
end

# TODO should this be removed?
function CUDA.CuArray(ten::CentralTensor)
    e11, e12 ,e21, e22 = CuArray.((ten.e11, ten.e12, ten.e21, ten.e22))
    @cast V[(u1, u2), (d1, d2)] := e11[u1, d1] * e21[u2, d1] *
                                   e12[u1, d2] * e22[u2, d2]
    V ./ maximum(V)
end

mutable struct DiagonalTensor{T <: Real, N} <: AbstractSparseTensor{T, N}
    e1 # ::MatOrCentral{T, N}
    e2 # ::MatOrCentral{T, N}
    dims::Dims{N}

    function DiagonalTensor(e1, e2)
        dims = (size(e1, 1) * size(e2, 1), size(e1, 2) * size(e2, 2))
        T = promote_type(eltype.((e1, e2))...)
        new{T, 2}(e1, e2, dims)
    end
end

function move_to_CUDA!(ten::DiagonalTensor)
    ten.e1 = move_to_CUDA!(ten.e1)
    ten.e2 = move_to_CUDA!(ten.e2)
    ten
end

device(ten::DiagonalTensor) = union(device(ten.e1), device(ten.e2))


mpo_transpose(ten::DiagonalTensor) = DiagonalTensor(mpo_transpose.((ten.e2, ten.e1))...)

mutable struct VirtualTensor{T <: Real, N} <: AbstractSparseTensor{T, N}
    con # ::MatOrCentral{T, 2}
    projs::Proj{6}  # == (p_lb, p_l, p_lt, p_rb, p_r, p_rt)
    dims::Dims{N}

    function VirtualTensor(con, projs::Proj{6})
        T = eltype(con)
        dims = (length(projs[2]), maximum(projs[3]) * maximum(projs[6]),
                length(projs[5]), maximum(projs[1]) * maximum(projs[4]))
        new{T, 4}(con, projs, dims)
    end
end

device(ten::VirtualTensor) = device(ten.con)

function move_to_CUDA!(ten::VirtualTensor)
    ten.con = move_to_CUDA!(ten.con)
    ten
end

mpo_transpose(ten::VirtualTensor) = VirtualTensor(ten.con, ten.projs[[3, 2, 1, 6, 5, 4]])
mpo_transpose(ten::Array{<:Real, 4}) = Array(permutedims(ten, (1, 4, 3, 2)))  # TODO CuArrayOrArray ???
mpo_transpose(ten::Array{<:Real, 2}) = Array(transpose(ten))  # TODO CuArrayOrArray ???

const SparseTensor{T, N} = Union{
    SiteTensor{T, N}, VirtualTensor{T, N}, CentralTensor{T, N}, DiagonalTensor{T, N}
}
const Tensor{T, N} = Union{CuArrayOrArray{T, N}, SparseTensor{T, N}}
const CentralOrDiagonal{T, N} = Union{CentralTensor{T, N}, DiagonalTensor{T, N}}

Base.eltype(ten::Tensor{T, N}) where {T <: Real, N} = T
Base.ndims(ten::Tensor{T, N}) where {T, N} = N
Base.size(ten::SparseTensor, n::Int) = ten.dims[n]
Base.size(ten::SparseTensor) = ten.dims
