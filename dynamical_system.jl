## Figures for the dynamical systems to get all eigenvectors.
# SS-HOPM finds
# lambda = 0.8730, 0.4306, 0.0180, 0.0006

using PyPlot
using Combinatorics
using StatsBase

# Alternative methods will need to be developed
# for finding the unstable real eigenpairs,
# i.e., eigenpairs for which C(λ, x) is indefinite.

function mult3(T::Array{Float64,3}, x::Vector{Float64})
    dims = size(T)
    M = zeros(dims[1],dims[2])
    for i=1:dims[3]
        M += T[:,:,i]*x[i]
    end
    return M
end

"""
T - the tensor
x0 - the starting vector, default is randn(size(T,1))
f - the filter function, default
"""
function dynsys_tensor_eigenvector(T; xinit=Void, f=x -> abs.(x), k=1, h=0.5, histlam=[])
    x = randn(size(T,1))
    if xinit!=Void
        x[:] = xinit
    end
    normalize!(x)

    # This is the ODE function
    F = function(x::Vector{Float64})
        M = mult3(T, x)
        d,V = eig(M)
        # filter to the real ones
        realf = abs.(imag(d)) .<= size(M,1)*eps(Float64)
        d = d[realf]
        V = V[:,realf]
        # now apply the filter function f and sort
        p = sortperm(f(d))
        v = V[:,p[k]] # pick out the kth eigenvector
        # normalize the sign
        if real(v[1]) >= 0
            v *= -1.0
        end
        return real(v) - x
    end

    # Forward Euler method
    for iter=1:100
        x = x + h*F(x)
        if eltype(histlam) == Float64
            quot = x'*mult3(T,x)*x;
            push!(histlam, quot[1])
        end
    end
    return x, x'*mult3(T,x)*x
end


#
function symtensor(T::Array{Float64})
    S = zeros(T)
    d = ndims(T)
    for p in permutations(1:d)
        S += permutedims(T,p)
    end
    S
end
##

# This exmaple is from Kolda and Mayo, Example 3.6
# It has 7 real eigenvalue pairs
function example_tensor()
    T = zeros(3,3,3)
    T[1,2,3] = -0.1790
    T[2,3,3] = 0.1773 / 2
    T[1,1,2] = 0.0516 / 2
    T[1,1,3] = -0.0954 /2
    
    T[1,2,2] = -0.1958 /2
    T[1,3,3] = -0.2676 /2
    T[2,2,2] = 0.3251 /6
    T[2,2,3] = 0.2513 /2
    T[3,3,3] = 0.0338 /6
    
    T = symtensor(T)
    T[1,1,1] = -0.1281
    T[2,2,2] = 0.3251
    T[3,3,3] = 0.0338
    return T
end

function results(T, f, ntrials; k=1)
    lams = zeros(ntrials)
    for t=1:ntrials
        temp = dynsys_tensor_eigenvector(T; f = f, k=k)[2]
        lams[t] = temp[1]
    end
    lams
end

function main()
    T = example_tensor()
    srand(1) # for consistent results
    ntrials = 5
    display(countmap(map(x -> round(x, 4), results(T, x -> -abs.(x), ntrials))))
    display(countmap(map(x -> round(x, 4), results(T, x -> abs.(x), ntrials))))
    display(countmap(map(x -> round(x, 4), results(T, x -> x, ntrials))))
    display(countmap(map(x -> round(x, 4), results(T, x -> -x, ntrials))))
    display(countmap(map(x -> round(x, 4), results(T, x -> x, ntrials; k=2))))

    # make plot
    histlam = zeros(0)
    srand(1)
    x, val = dynsys_tensor_eigenvector(T; f = x -> x, k=2, histlam=histlam)

    PyPlot.pygui(true)
    plot(1:30, histlam[1:30])
    xlabel("Iteration")
    ylabel("Rayleigh quotient")
    title("lambda = 0.2294")
    show()
    savefig("convergence.eps")
end

main()
