include("unfolding.jl")
using Plots
A = zeros(3,3,3)

for i = 1:length(size(A))
    A[i,i,i] = 3
end

I = eye(3)
function M(x)
    P = unfold(A,1)*(kron(x,I))
    return P*x
end

function jacobian(f,x)
n=length(x);
J = zeros(n,n);
fx=f(x)
eps=1.e-8; 
xperturb=x;
    for i=1:n
        xperturb[i]=xperturb[i]+eps;
        J[:,i]=(f(xperturb)-fx)/eps;
        xperturb[i]=x[i];
    end;
return J
end

function euler(f, x0, timesteps, h)
    data = zeros(length(x0),timesteps)
    data[:,1] = x0
    for i = 2:timesteps
        data[:,i] = data[:,i-1] + h*f(data[:,i-1]);
    end
    return data
end


x0 = rand(3,1)
timesteps = 1000;
h = 0.01;
data =euler(M,x0,timesteps,h)
plt = plot(data[1,:])

