### local_startup.jl is loaded at startup by:  ~/.julia/config/startup.jl


using LinearAlgebra, RowEchelon, Latexify, Combinatorics

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

# Matrix Utils
Lx = latexify # function
const var"@Lx" = var"@latexify" # macro: @Lx A*x = 0
Cx = collect # Used in Latex to force better rendering
cv(m,i) = m[:,i]             # col vector i of m
rv(m,i) = collect(m[i,:]')   # row vector i of m, in col format


