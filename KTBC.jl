using LinearAlgebra, ToeplitzMatrices, RowEchelon
import LinearAlgebra.ldlt

# Create an identity matrix
eye(n) = Matrix{Int64}(I, n, n)
eye(T,n) = map(T,eye(n))

# Set diagonal (of offset) to val. offset=1 is above and offset-1 is below
function setdiag!(m,offset,val)
    n = size(m)[1]
    ds = diag(reshape(collect(1:n^2), n, n), offset)
    setindex!(m, val*ones(n - 1), ds)
    m
end

# Symmetric Toeplitz matrix - FIXED for Julia 1.11
k_toeplitz(n) = Int64.(Matrix(SymmetricToeplitz([2; -1; zeros(n-2)])))

# K Matrix
function K_n(n)
    k_toeplitz(n)
end

# T means Top Matrix
function T_n(n)
    m = k_toeplitz(n)
    m[1,1] = 1
    m
end

# B means Bottom Matrix
function B_n(n)
    m = k_toeplitz(n)
    m[1,1] = 1
    m[n,n] = 1
    m
end

# C means Circular or Convolution Matrix
function C_n(n)
    m = k_toeplitz(n)
    m[1,n] = -1
    m[n,1] = -1
    m
end

# Difference Matrix [1, -1, 0, 0; 0, 1, -1, 0]
U_n(n) = setdiag!(eye(n), 1,-1)

# Summation Matrix [1, 1, 1; 0, 1, 1; 0, 0, 1]
S_n(n) = Int64.(inv(U_n(n)))

# First Order Forward Difference Matrix [-1, 1, 0, 0; 0, -1, 1, 0]
function F_n(n)
    m = -eye(n)
    setdiag!(m,1,1)
    m
end

# First Order Reverse (Backward) Difference Equation [1, 0, 0, 0; -1, 1, 0, 0]
function R_n(n)
    m = eye(n)
    setdiag!(m,-1,-1)
    m
end

# Elimination Matrix: subtract val*from_row from to_row
function E_n(n,to_row,from_row,val) # nxn matrix
    m = Matrix{Float64}(I, n, n)
    m[to_row,from_row] = -val
    m
end

# P_n(n,i,j)*A permutes ith row with jth row
# A*P_n(n,i,j) permutes ith col with jth col
function P_n(n,i,j)
    m = eye(n)
    m[i,j], m[j,i] = 1, 1
    m[i,i], m[j,j] = 0, 0
    m
end

# P_n(idx) generates P each row containing 1 according to the idx
function P_n(idx)
    dim = length(idx)
    @assert sort(idx) == 1:dim "$idx should have the exact the elements of 1:$dim"
    m = zeros(Int64,dim,dim)
    for (i,j)=zip(1:dim,idx)
        m[i,j] = 1
    end
    m
end

# LDL' Factorization where S must be symmetric
function ldlt(S::Matrix)
    F = ldlt(SymTridiagonal(S))
    Matrix(F.L), Matrix(F.D), Matrix(F.L')
end

# CR Factorization
function cr(A::Matrix)r
    r, p = rref_with_pivots(A)
    r  = [x' for x=eachrow(r) if sum(x) != 0]
    r  = reduce(vcat, r)
    A[:, p], round.(r,digits=10)
end

# Vectors - FIXED for Julia 1.11
constant(c,n) = c * ones(n)
linear(n)     = [i for i=1:n]
squares(n)    = [i^2 for i=1:n]
cubes(n)      = [i^3 for i=1:n]
delta(k,n)    = [zeros(k-1); 1; zeros(n-k)]      # delta(3,5)' = [0, 0, 1, 0, 0]
step(k,n)     = [zeros(k-1); 1; ones(n-k)]       # step(3,5)'  = [0, 0, 1, 1, 1]
ramp(k,n)     = [zeros(k-1); collect(0:(n-k))]   # ramp(3,5)'  = [0, 0, 0, 1, 2]
sines(t,n)    = [sin(t*i) for i=0:n-1]
cosines(t,n)  = [cos(t*i) for i=0:n-1]
exps(t,n)     = [exp(im*t*i) for i=0:n-1]


# Example: Different style to work with matrix indexes
function dr(A::Matrix)  # Added type annotation for clarity
    r, p = rref_with_pivots(A)
    z = map(x-> sum(x)==0, eachrow(r))  # is sum(row) == 0
    z = findall(z)                      # idxs of rows with all zeros
    r = r[setdiff(1:end, z), :]         # setdiff excludes z from 1:end
    A[:,p], round.(r,digits=10)
end
