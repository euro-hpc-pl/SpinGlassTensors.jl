
for (N, T) ∈ ((:QMps, :TensorMap), (:QMpo, :NestedTensorMap))
    @eval begin
        export $N
        struct $N{T <: Number} <: AbstractTensorNetwork
            tensors::$T
            sites::Vector{Site}

            $N(ten) = new{eltype(tensors)}(ten, sort(collect(keys(ten))))
        end
    end
end

function local_dims(mpo::QMpo, dir::Symbol)
    @assert dir ∈ (:down, :up)
    lds = Dict{Site, Int}()
    for site ∈ mpo.sites
        mkeys = sort(collect(keys(mpo[site])))
        if any(length(size(mpo[site][k])) > 2 for k ∈ mkeys)
            if dir == :down
                ss = size(mpo[site][last(mkeys)])
                push!(lds, site => length(ss) == 4 ? ss[4] : ss[2])
            else
                ss = size(mpo[site][first(mkeys)])
                push!(lds, site => length(ss) == 4 ? ss[2] : ss[1])
            end
        end
    end
    lds
end

# TODO this should be defined by the action of I
function IdentityQMps(::Type{T}, loc_dims::Dict{Site, Int}, Dmax::Int=1) where T
    id = Dict{Site, Array{T, 3}}(
        site => zeros(T, Dmax, ld, Dmax) for (site, ld) ∈ loc_dims
    )
    site_min, ld_min = minimum(loc_dims)
    site_max, ld_max = maximum(loc_dims)

    if site_min == site_max
        id[site_min] = zeros(T, 1, ld_min, 1)
    else
        id[site_min] = zeros(T, 1, ld_min, Dmax)
        id[site_max] = zeros(T, Dmax, ld_max, 1)
    end

    for (site, ld) ∈ loc_dims
        id[site][1, :, 1] .= one(T) / sqrt(ld)
    end
    QMps(id)
end

function Base.rand(::Type{QMps{T}}, sites::Vector, D::Int, d::Int) where T <: Real
    QMps = Dict{Site, Array{T, 3}}()
    for i ∈ sites
        if i == 1
            push!(QMps, i => randn(T, 1, d, D))
        elseif i == last(sites)
            push!(QMps, i => randn(T, D, d, 1))
        else
            push!(QMps, i => randn(T, D, d, D))
        end
    end
    QMps(QMps)
end

function Base.rand(
    ::Type{QMpo{T}}, sites::Vector, D::Int, d::Int, sites_aux::Vector=[], d_aux::Int=0
) where T <: Number
    QMpo = Dict{Site, Dict{Site, Array{T, 4}}}()
    QMpo_aux = Dict{Site, Array{T, 4}}()

    for i ∈ sites
        if i == 1
            push!(QMpo_aux, i => randn(T, 1, d, d, D))
            push!(QMpo_aux, (j => randn(T, d_aux, d_aux) for j ∈ sites_aux)...)
            push!(QMpo, i => copy(QMpo_aux))
        elseif i == last(sites)
            push!(QMpo_aux, (j => randn(T, d_aux, d_aux) for j ∈ sites_aux)...)
            push!(QMpo_aux, last(sites) => randn(T, D, d, d, 1))
            push!(QMpo, i => copy(QMpo_aux))
        else
            push!(QMpo_aux, (j => randn(T, d_aux, d_aux) for j ∈ sites_aux)...)
            push!(QMpo_aux, i => randn(T, D, d, d, D))
            push!(QMpo, i => copy(QMpo_aux))
        end
        empty!(QMpo_aux)
    end
    QMpo(QMpo)
end
