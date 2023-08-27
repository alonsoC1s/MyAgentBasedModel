using Pkg
Pkg.activate(".")

using MyAgentBasedModel

omp = OpinionModelProblem((-4, 4), (-4, 4))

solve(omp)