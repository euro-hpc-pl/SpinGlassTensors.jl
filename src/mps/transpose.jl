
function Base.transpose(mpo::QMpo{T}) where T <: Real
    QMpo(MpoTensorMap{T}(keys(mpo.tensors) .=> mpo_transpose.(values(mpo.tensors))))
end

mpo_transpose(M::MpoTensor{T, 2}) where T <: Real = M

function mpo_transpose(M::MpoTensor{T, 4}) where T <: Real
    MpoTensor{T, 4}(
        mpo_transpose.(reverse(M.bot)),
        mpo_transpose(M.ctr),
        mpo_transpose.(reverse(M.top)),
        M.dims[[1, 4, 3, 2]]
    )
end
