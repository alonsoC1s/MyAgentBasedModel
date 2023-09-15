using Pkg
Pkg.activate(".")

using MyAgentBasedModel
using JLD2

# omp = OpinionModelProblem((-2, 2), (-2, 2))
# Loading the exact same initial state
init_state = load("../new_settings.jld2")
X, Y, Z, A, B, C = init_state["X"], init_state["Y"], init_state["Z"], init_state["A"], init_state["B"], init_state["C"]
p = OpinionModelParams()

omp = OpinionModelProblem(p, X, Y, Z, C, A, B)

rX, rY, rZ, rC = solve(omp)

jldsave("../stochastic_new.jld2"; rX)