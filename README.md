[![Coverage Status](https://coveralls.io/repos/github/iitis/SpinGlassTensors.jl/badge.svg?branch=master)](https://coveralls.io/github/iitis/SpinGlassTensors.jl?branch=master)
# SpinGlassTensors.jl
Basic structures, like an `MPS` and `MPO` for various usages, predominantly in approximation of low energy spectrums of Ising instances.

## Motivation
This library provides structure like matrix product states and matrix product operators along with basic algorithms for manipulating them.


## Usage

Some examples of usage are

- `MPO`-`MPS` contractions
```julia
using SpinGlassTensors

D = 32
d = 2
sites = 100

ψ = randn(MPS{T}, sites, D, d)
mpo_ψ = randn(MPO{T}, sites, D, d)
ϕ = mpo_ψ * ψ
```
- `MPS` compression
```julia
D = 32
Dcut = 16

d = 2
sites = 100
ψ = randn(MPS{T}, sites, D, d)
truncate!(ψ, :right, Dcut)
```
