export
    measure_memory,
    format_bytes

measure_memory(ten::AbstractArray) = [Base.summarysize(ten), 0]
measure_memory(ten::CuArray) = [0, prod(size(ten)) * sizeof(eltype(ten))]
measure_memory(ten::Diagonal) = measure_memory(diag(ten))
measure_memory(ten::SiteTensor) = sum(measure_memory.([ten.loc_exp, ten.projs...]))
measure_memory(ten::CentralTensor) = sum(measure_memory.([ten.e11, ten.e12, ten.e21, ten.e22]))
measure_memory(ten::DiagonalTensor) = sum(measure_memory.([ten.e1, ten.e2]))
measure_memory(ten::VirtualTensor) = sum(measure_memory.([ten.con, ten.projs...]))
measure_memory(ten::MpoTensor) = sum(measure_memory.([ten.top..., ten.ctr, ten.bot...]))
measure_memory(ten::Union{QMps, QMpo}) = sum(measure_memory.(values(ten.tensors)))
measure_memory(env::Environment) = sum(measure_memory.(values(env.env)))

function format_bytes(bytes, decimals::Int = 2, k::Int = 1024)
    bytes == 0 && return "0 Bytes"
    dm = decimals < 0 ? 0 : decimals
    sizes = ["Bytes", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
    i = convert(Int, floor(log(bytes) / log(k)))
    string(round((bytes / ^(k, i)), digits=dm)) * " " * sizes[i+1]
end
