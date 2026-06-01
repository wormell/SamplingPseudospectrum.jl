# SamplingPseudospectrum.jl

[![CI](https://github.com/wormell/SamplingPseudospectrum.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/wormell/SamplingPseudospectrum.jl/actions/workflows/ci.yml)

This package implements the algorithm to compute the sampling pseudospectrum $\hat P(λ)$ as proposed in [Wormell 2026](https://arxiv.org/abs/2605.15234).

In the case of *i.i.d.* processes (i.e. no lag computations) it can be used to compute $P(λ)$ as well, by inputting interpolation nodes and weights as data.

Sample code:

```julia
using SamplingPseudospectrum

f(x) = 3.8x*(1-x) #dynamics
N = 6; M = 1000
L = 10 # lag time for correlations

# create an ergodic sample
x = rand()
for i = 1:1000; x = f(x); end # initialising sample
xh = Array{Float64}(undef,M+1)
xh[1] = x
for i = 1:M; xh[i+1] = f(xh[i]); end

# use polynomial dictionary
ΨX = transpose(xh[1:end-1]).^(0:N-1)
ΨY = transpose(xh[2:end]).^(0:N-1)

λs = (-1.1:0.05:1.1) .+ (0:0.05:1.1)'*im
Pestimate(λs,ΨX,ΨY;L,normalized=true) #outputs an array of MP̂(λ) at λs
```
