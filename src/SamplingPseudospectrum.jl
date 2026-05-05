module SamplingPseudospectrum

using LinearAlgebra, DSP, OffsetArrays
import OffsetArrays: no_offset_view

export Pestimate, getlagweights

"""
    spectralextrema(A,B)

Return top and bottom generalised eigenvalues of matrices `A` and `B`, which are assumed to be Hermitian """
function spectralextrema(A,B)
    evs = real(eigvals(Hermitian(A),Hermitian(B),sortby=abs))
    evs[1], evs[end]
end

"""
    lagfunction(r)

Return the positive definite kernel κ(`r`) with κ(0)=1 and supp κ = [-1,1], having minimal -κ''(0).
"""
lagfunction(r) = begin ar = abs(r); ar < 1 ? lagfunction_unsafe(ar) : zero(r); end
lagfunction_unsafe(r) = sinpi(r)/pi + (1-r)*cospi(r) 

"""
    getlagweights(L,samplingeigs=nothing)

Provide a positive definite weight function for input to the sampling pseudospectrum 
algorithm, with basic windowing length `L` plus cancellation of resonances given by `samplingeigs`. 

`samplingeigs` must contain complex conjugates, or it will return an error.
"""
getlagweights(L,samplingeigs::Nothing=nothing) = OffsetArray(lagfunction.((-L:L)/(L+1)),-L:L)
function getlagweights(L,samplingeigs)
    LS = max(0,L+length(samplingeigs)-1)
    # convolving with the sampling eigenvalues will smear the eigenvectors that aren't picked up across time
    # this adjustment probably needs to be actually thought about
    
    lagweights = ones(eltype(samplingeigs),1)
    for μ in samplingeigs
        @assert !(μ ≈ 1)
        lagweights = conv(lagweights,[1,-μ]/(1-μ))
    end
    if !(all(lagweights .≈ real.(lagweights))) # if you remove this remember to put conjugates below in every time you reverse
        error("samplingeigs do not contain complex conjugates.") 
    else
        lagweights = real(lagweights)
    end
    
    lagweights = conv(lagweights,reverse(lagweights))
    
    lagweights = conv(lagweights,lagfunction.((-LS:LS)/(LS+1)))
    LL = div(length(lagweights)-1,2)
    OffsetArray(lagweights,-LL:LL)
end


function _Piterloop(ΨX,ΨRW,ΨXR,Ares,lagweights::OffsetArray)
    N,M = size(ΨX)
    @assert -firstindex(lagweights) == lastindex(lagweights)
    L = lastindex(lagweights);
    tailweights = cumsum(lagweights[begin:-1])

    Anew = zeros(eltype(Ares),N,N)
    for m = 1:M
        AX = Ares * view(ΨX,:,m)

        UXAX = dot(view(ΨX,:,m),AX)*lagweights[0]/2M
        RUXAX = UXAX*view(ΨRW,:,m)
        for l = #max(-L,1-m)
            1:min(L,M-m)
            ml = m+l
            UXAX = dot(view(ΨX,:,ml),AX)*lagweights[l]/M
            axpy!(UXAX,view(ΨRW,:,ml),RUXAX)
        end
        if m<=L
            mul!(RUXAX, ΨXR',AX, tailweights[L-m+1]/M, 1)
        elseif M-m+1<=L
            mul!(RUXAX, ΨXR',AX, tailweights[L-(M-m+1)+1]/M, 1)
        end
        Anew .+= RUXAX .* view(ΨRW,:,m)'
    end
    
    axpy!(-sum(no_offset_view(lagweights) .* (1 .+abs.(-L:L)/M))/2, ΨXR'*Ares*ΨXR, Anew) # TODO: should this be 1 .- abs.(-L:L)/M ?
    Anew .+= Anew'
    Anew
end

"""
   roundtol(x,tol) 

Print `x` rounded to the same number of digits as the leading digit of `tol`.
"""
roundtol(x,tol) = round(x,sigdigits=floor(Int,-log10(tol))) # for printing convergence

weightbyw!(ΨR,W::UniformScaling{T}) where T = rmul!(ΨR,W.λ')
weightbyw!(ΨR,W::UniformScaling{Bool}) = W.λ ? ΨR : fill!(ΨRW,0)
weightbyw!(ΨR,W::Union{Diagonal,Tridiagonal,SymTridiagonal}) = rmul!(ΨR,W')
function weightbyw!(ΨR,W) 
    ΨR .= ΨR * W'
    ΨR
end

"""
    Pestimate(λ, ΨX, ΨY; W=I, tol=0.1, maxiter=50, 
           L=0, samplingeigs=nothing, lagweights=getlagweights(lag,samplingeigs),
        normalized=false, verbose=false
        )

Compute P̂(`λ`) from data `ΨX` and `ΨY`, with covariance weighting `W`.

Applying a vector of `λ` allows the algorithm to reuse leading matrices, reducing the
number of iterations required.

# Arguments
- `λ`: the eigenvalue(s) at which to compute the problem
- `ΨX`: "initial" data, with snapshots stored as columns
- `ΨY`: "final" data, with snapshots stored as columns
- `W`: matrix of weights of different sample points
- `tol`: certified relative tolerance for computing P̂
- `maxiter`: maximum number of iterations to attempt
- `lagweights`: weights for lag-correlations in computation of Birkhoff variance
- `L`, `samplingeigs`: input to `getlagweights` function when computing `lagweights`
- `normalized`: if true, return M P̂(λ), where M is the number of snapshots. This is helpful to test for eigenvalues
- `verbose`: print progress of iterations
"""
function Pestimate(λs::Array,ΨX, ΨY; W=I, 
            tol=0.1, maxiter=50, verbose=false, 
           L::Integer=0, samplingeigs=nothing,
        lagweights=getlagweights(L,samplingeigs),
        normalized=false
        )
    T = promote_type(eltype(ΨX),eltype(ΨY),eltype(λs),eltype(lagweights))
    Pλs = real(similar(λs))
    N,M = size(ΨX)
    @assert size(ΨY) == size(ΨX)

    normalisation_constant = normalized ? M : 1

    
    # # Start by orthonormalising ΨX to maximise numerical stability
    pS,pU = eigen(Hermitian(ΨX * (W*ΨX') / M))
    ΨX = Diagonal(sqrt.(pS))\(pU'*ΨX)
    ΨY = Diagonal(sqrt.(pS))\(pU'*ΨY)

    ΨXX::Matrix{T} = ΨX * (W*ΨX') / M
    ΨXY::Matrix{T} = ΨX * (W*ΨY') / M

    ΨXR::Matrix{T} = Array{T}(undef,N,N)
    ΨRW::Matrix{T} = Array{T}(undef,N,M)
    A::Matrix{T} = Array{T}(I/sqrt(N),N,N) # norm(A) = 1, importantly

    for (λn,λ) in enumerate(λs)
        A /= norm(A)
        ΨRW .= conj(λ) .* ΨX .- ΨY; weightbyw!(ΨRW,W)
        
        ΨXR .= λ*ΨXX - ΨXY
        U,S,V = svd(ΨXR)
        Σ = Diagonal(S)
        if minimum(S) < 100eps(maximum(S))
            Pλs[λn] = 0
            verbose && println("λ = $λ: apparent eigenvalue, setting P(λ) = 0")
            continue
        end
            # println("ΨXR svd extrema = ",extrema(S))

        eigA = Inf
        for i = 1:maxiter
            Ares = (U*(Σ\(V'*A*V)/Σ)*U')
            @assert norm(Ares-Ares').< 1000eps()*norm(Ares) # else numerical error has Gone Mad
            Ares .= (Ares + Ares')/2
            @assert isposdef(Ares) # or we're already in trouble!

            Anew = _Piterloop(ΨX,ΨRW,ΨXR,Hermitian(Ares),lagweights)
            
            eigA = norm(Anew)
            eigAmin,eigAmax = spectralextrema(Anew,A)
            condn = eigAmax/eigAmin
            verbose && println("P($λ) bounds = [$(roundtol(normalisation_constant/eigAmax,tol)),$(roundtol(normalisation_constant/eigAmin,tol))], relative error = $(round(condn-1,sigdigits=2))")

            if 1 <= condn < 1+tol
                verbose && println("λ = $λ: $i iterations")
                break
            end
            A = Anew / eigA
            i==maxiter && verbose && println("λ = $λ: maxiter=$i iterations")
        end
        Pλs[λn] = normalisation_constant/eigA
    end
    return Pλs
end
Pestimate(λ::Number,ΨX, ΨY; kwargs...) = Pestimate([λ],ΨX, ΨY; kwargs...)[1]


end # module SamplingPseudospectrum
