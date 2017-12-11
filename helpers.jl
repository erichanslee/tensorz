using PyPlot
using Combinatorics
using StatsBase
using DifferentialEquations

NUMITERS = 200

# Alternative methods will need to be developed
# for finding the unstable real eigenpairs,
# i.e., eigenpairs for which C(λ, x) is indefinite.
function symtensor(T::Array{Float64})
    S = zeros(T)
    d = ndims(T)
    for p in permutations(1:d)
        S += permutedims(T,p)
    end
    S
end
##

# Binary 2x2x2 Tensor
function bintensor(x)

    # x is a length 4 vector whos entries are either 1 or 0
    T = zeros(2,2,2)
    T[1,1,1] = x[1]
    T[1,1,2] = x[2]
    T[1,2,1] = x[2]
    T[2,1,1] = x[2]
    T[1,2,2] = x[3]
    T[2,2,1] = x[3]
    T[2,1,2] = x[3]
    T[2,2,2] = x[4]
    return T
end


# Creates a rank k 3rd order orthogonal Tensor of dimension d with linear spectra
# Returns the Tensor T and Eigenvectors X
function orthtensor(d,k)

    function rank1(x)
        d = length(x);
        T = zeros(d,d,d)
        for i = 1:d
            for j = 1:d
                for k = 1:d
                    T[i,j,k] = x[i]*x[j]*x[k];
                end
            end
        end
        return T
    end

    T = zeros(d,d,d)
    X,R = qr(rand(d,k))
    print(X)

    for i = 1:k
        T = T + i*rank1(X[:,k]); # Spectra should be 1:d
    end

    return X, T

end


# This exmaple is from Kolda and Mayo, Example 3.6
# It has 7 real eigenvalue pairs
function example_tensor3()
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


function example_tensor4()
    T = zeros(3,3,3,3)
    T[1,1,2,3] = -0.2939/2
    T[1,2,3,3] = 0.0919 /2
    T[2,2,3,3] = 0.2127/4
    T[1,1,1,2] = -0.0031/6
    T[1,1,3,3] = 0.3847/4
    T[1,3,3,3] = -0.3619/6
    T[2,3,3,3] = 0.2727/6
    T[1,1,1,3] = 0.1973/6
    T[1,2,2,2] = 0.2972/6 
    T[1,1,2,2] = -0.2485/4
    T[1,2,2,3] = 0.1862/2
    T[2,2,2,3] = -0.3420/6
    T = symtensor(T)
    
    T[1,1,1,1] = 0.2883 
    T[2,2,2,2] = 0.1241 
    T[3,3,3,3] = -0.3054
    return T
end

function random_tensor4()
    T = zeros(3,3,3,3);
    T = symtensor(T);
    for i = 1:3
        for j = 1:3
            for k = 1:3
                for l =1:3
                    T[i,j,k,l] = rand();
                end
            end
        end
    end
    return T;
end

function mult3(T::Array{Float64,3}, x::Vector{Float64})
    dims = size(T)
    M = zeros(dims[1],dims[2])
    for i=1:dims[1]
        for j=1:dims[2]
            for k=1:dims[3]
                M[i,j] += T[i,j,k]*x[k]
            end
        end
    end
    
    return M
end

function mult4(T::Array{Float64,4}, x::Vector{Float64})
    dims = size(T)
    M = zeros(dims[1],dims[2])
    for i=1:dims[1]
        for j=1:dims[2]
            for k=1:dims[3]
                for l=1:dims[4]
                    M[i,j] += T[i,j,k,l]*x[k]*x[l]
                end
            end
        end
    end
    
    return M
end

function jacobian(f,x)
    m=length(x);
    J = zeros(m,m);
    fx=f(x);
    eps=1.e-6;
    xperturb=x;
    for i=1:m
       xperturb[i]=xperturb[i]+eps;
       J[:,i]=(f(xperturb)-fx)/eps;
       xperturb[i]=x[i];
    end
return J;
end

function MPowerIteration(T; f=x -> abs.(x), k=1,  histlam=[])
    x = 5*randn(size(T,1))
    
    F = function(x::Vector{Float64})
        M = mult4(T, x)
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
        return v;
    end

    # Main Power Iteration Loop
    for i = 1:NUMITERS
        v = F(x)
        x = v;
        # Update Data
        if eltype(histlam) == Float64
            quot = x'*mult4(T,x)*x;
            push!(histlam, quot[1])
        end
    end

    J = jacobian(F,x)
    D,V = eig(J)
    println("Ending Jacobian Eigenvalues:")
    println(D)

    return x, x'*mult4(T,x)*x

end

"""
T - the tensor
x0 - the starting vector, default is randn(size(T,1))
f - the filter function, default
"""
function dynsys_tensor_eigenvector(T; xinit=Void, f=x -> abs.(x), k=1, h=0.5, histlam=[], multiplier=1)
    x = 5*randn(size(T,1))
    if xinit!=Void
        x[:] = xinit
    end
    normalize!(x)



    # This is the ODE function
    F = function(x::Vector{Float64})
        M = mult4(T, x)
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
        return multiplier*(real(v) - 10*x)
    end

    J = jacobian(F,x)
    D,V = eig(J)
    println("Starting Jacobian Eigenvalues:")
    println(D)

    # Forward Euler method
    for iter=1:NUMITERS
        x = x + h*F(x)
        if eltype(histlam) == Float64
            quot = x'*mult4(T,x)*x;
            push!(histlam, quot[1])
        end
    end

    # # Use ODE Solver
    # tspan = (0.0, 5.0)
    # ff(t,y) = F(y)
    # prob = ODEProblem(ff,x,tspan)
    # sol = solve(prob,Tsit5())
    # x = sol[:,end]

    J = jacobian(F,x)
    D,V = eig(J)
    println("Ending Jacobian Eigenvalues:")
    println(D)

    return x, x'*mult4(T,x)*x
end
