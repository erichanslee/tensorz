
function hyper4test()

    function rank1(x)
        d = length(x);
        T = zeros(d,d,d,d)
        for i = 1:d
            for j = 1:d
                for k = 1:d
                    for l = 1:d
                        T[i,j,k,l] = x[i]*x[j]*x[k]*x[l];
                    end
                end
            end
        end
        return T
    end

    
    #edge set contains the number of different summands
    #in the higher order laplacian
    edgeset = 
    [
    1 2;
    1 3;
    1 4;
    1 5;
    1 6;
    2 3;
    2 4;
    2 5;
    2 6;
    3 4;
    3 5;
    3 6;
    4 5;
    4 6;
    5 6
    ];

    # n is the length of edge set
    n = 15;
    dim = 6;

    T = zeros(dim,dim,dim,dim);
    for k = 1:n
        idx1 = edgeset[k,1];
        idx2 = edgeset[k,2];
        eij = zeros(dim);
        eij[idx1] = 1;
        eij[idx2] = -1;
        T = T + rank1(eij);
    end

    return T;

ennd
end
