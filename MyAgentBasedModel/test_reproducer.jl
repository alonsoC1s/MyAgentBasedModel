using Pkg

# Pkg.activate(".")

using MyAgentBasedModel
using JLD2

# cd("~/Dokumente/MyAgentBasedModel/MyAgentBasedModel/")

name = "fixed" * "_n.jld2"

# omp = OpinionModelProblem((-2, 2), (-2, 2))
# Loading the exact same initial state
init_state = load("../test_data/new_settings.jld2")
X, Y, Z, A, B, C = init_state["X"], init_state["Y"], init_state["Z"], init_state["A"], init_state["B"], init_state["C"]


#=
    Zero noise, everything else the same
=# 
# p = OpinionModelParams(4, 2, 250, 15, 1, 2, 4, 0.0, 0, 0, 10, 100)
p = OpinionModelParams()

omp = OpinionModelProblem(p, X, Y, Z, C, A, B)

X, Y, Z, C = solve(omp)

jldsave("../test_data/" * name; X, C, B)