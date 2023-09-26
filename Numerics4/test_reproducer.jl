using JLD2
using Pkg

# cd("~/Dokumente/MyAgentBasedModel/Numerics4")
Pkg.activate(".")

name = "fixed" * "_l.jld2"

include("src/abm.jl") # Gives access to ABMsolve & plotting

# q = parameters(sigma = 0.0, sigmahat = 0.0, sigmatilde = 0.0)
X, C, infs, meds, state, (p,q) = better_solve(chosenseed=260923)

jldsave("../test_data/" * name; X, C)