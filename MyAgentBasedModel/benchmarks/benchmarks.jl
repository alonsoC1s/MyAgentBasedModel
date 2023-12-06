using BenchmarkTools
using MyAgentBasedModel

include("../../Numerics4/src/abm.jl")

OpinionModelProblem((-1, 1), (-1, 1))
test_ambInit()

# Comparing the ABM initializers
new_omp = @benchmark OpinionModelProblem((-1, 1), (-1, 1))
old_omp = @benchmark test_ambInit()

println(judge(median(new_omp), median(old_omp)))