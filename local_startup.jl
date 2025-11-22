### local_startup.jl is loaded at startup by:  ~/.julia/config/startup.jl


using LinearAlgebra, RowEchelon, Latexify, Combinatorics, GenericLinearAlgebra

## This extends in to support substring matching for strings, giving you Python-style syntax
Base.in(needle::AbstractString, haystack::AbstractString) = occursin(needle, haystack)

## Custom LaTeX completions for Julia REPL
try
    using REPL

    # Add custom LaTeX symbols
    REPL.REPLCompletions.latex_symbols["\\grad"] = "∇"
    REPL.REPLCompletions.latex_symbols["\\curl"] = "×"
    REPL.REPLCompletions.latex_symbols["\\dive"] = "⋅"
    REPL.REPLCompletions.latex_symbols["\\del"] = "∂"
    REPL.REPLCompletions.latex_symbols["\\comp"] = "∘"
catch
    # Silently fail if REPL not available
end

Lx = latexify # function
const var"@Lx" = var"@latexify" # macro: @Lx A*x = 0
Cx = collect
## Matrix Utils

# col vector i of m
cv(m,i) = m[:,i]

# row vector i of m, in col format
rv(m,i) = collect(m[i,:]') 


# Normalize each column of A
normcols(A) = A ./ sqrt.(diag(A'A))'


"""
    isbasiseq(A, B; tol=1e-10)

Test if two bases have matching column directions (up to scaling and permutation).
Requires both matrices to be full rank.
"""
function isbasiseq(A, B; tol=1e-10)
    An = normcols(A)
    Bn = normcols(B)
    C = abs.(An' * Bn)
    M = C .> 1 - tol                                                                                                                                                                                             
    all(sum(M, dims=1) .== 1) && all(sum(M, dims=2) .== 1)
end

# Is A a Normal Matrix
isnormal(A) = A'A == A*A'

# Is A an Orthogonal Matrix
isorthogonal(A) = A'A == I

# Does A have Orthogonal Columns
isorthocols(A) = isdiag(A'A)

# Is A a Symmetric Matrix
issymmetric(A) = S==S'
