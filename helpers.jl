using Combinatorics
using StatsBase

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

function symtensor(T::Array{Float64})
    S = zeros(T)
    d = ndims(T)
    for p in permutations(1:d)
        S += permutedims(T,p)
    end
    S
end

function unfold(A, i)
    # dims not unfolded
    N = length(size(A));
    j = setdiff(1:N,i) ;

    # Permute dims of A, bring unfolded dimension to front
    B = permutedims(A,vcat(i,j))

    # size of unfolded dimension
    n = size(B,1)
    
    return reshape(B,(n,div(length(B),n)))
end

A = rand(3,3,3)

function funcM(x)
    dims = size(A)
    M = zeros(dims[1],dims[2])
    for i=1:dims[3]
        M += A[:,:,i]*x[i]
    end
    d,V = eig(M)
    # filter to the real ones
    realf = abs.(imag(d)) .<= size(M,1)*eps(Float64)
    d = d[realf]
    V = V[:,realf]
    # now apply the filter function f and sort
    p = sortperm(abs.(d))
    v = V[:,p[1]] # pick out the first eigenvector
    # normalize the sign
    if real(v[1]) >= 0
        v *= -1.0
    end
    return real(v) - x
end


function M(t,x)
    return funcM(x)
end

function jacobian(f,x)
m=length(x);
J = zeros(m,m);
fx=f(x);
eps=1.e-8;
xperturb=x;
for i=1:m
   xperturb[i]=xperturb[i]+eps;
   J[:,i]=(f(xperturb)-fx)/eps;
   xperturb[i]=x[i];
end
return J;
end

function euler(f, x0, timesteps, h)
    data = zeros(length(x0),timesteps);
    eigt = zeros(length(x0),timesteps);
    data[:,1] = x0;
    for i = 2:timesteps
        data[:,i] = data[:,i-1] + h*f(i,data[:,i-1]);;
        
    end
    return data;
end

function eulerJac(f, x0, timesteps, h)
    data = zeros(length(x0),timesteps)
    eigv = zeros(length(x0),timesteps)
    data[:,1] = x0
    for i = 2:timesteps
        data[:,i] = data[:,i-1] + h*f(data[:,i-1]);
        J = jacobian(f, data[:,i]);
        D,V = eig(J);
        eigv[:,i] = D;
    end
    return data, eigv;
end


