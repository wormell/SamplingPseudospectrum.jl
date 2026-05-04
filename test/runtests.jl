println("Testing...")
using Random, Statistics, SamplingPseudospectrum
RNG = MersenneTwister(0)

# Test 1: N = 1 closed form
M1 = 20
ΨXT1 = randn(RNG,1,M1)
ΨYT1 = randn(RNG,1,M1)
λsT1 = randn(RNG,10) + im*randn(RNG,10)
PET1 = Pestimate(λsT1, ΨXT1, ΨYT1; tol=1e-12, maxiter=10, normalized=false,verbose=false)
ΨXXT1 = (ΨXT1 * ΨXT1')[1,1]/M1
ΨXYT1 = (ΨXT1 * ΨYT1')[1,1]/M1

CsT1 = [λ*ΨXXT1-ΨXYT1 for λ in λsT1]
VCsT1sq = [mean(abs2.((λ*conj(ΨXT1)-conj(ΨYT1)).*ΨXT1)) for (λ,C) in zip(λsT1,CsT1)]
@assert all(PET1 .≈ abs2.(CsT1)./(VCsT1sq - abs2.(CsT1)))

println("Tests finished")