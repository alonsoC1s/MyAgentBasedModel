using MyAgentBasedModel
using JLD2

for selectedseed = [1, 22111998, 420, 43]
    # Generate a new opinonmodelproblem
    omp = OpinionModelProblem((-2, 2), (-2, 2))

    # Solve system, get evolution, compute rate tensor and save
    rX, rY, rZ, rC, rR = solve(omp; seed=selectedseed)

    rates_tensor = time_rate_tensor(rR, rC)

    # Save in JLD2 with the seed as name
    filename = "ratemat_seed" * string(selectedseed) * ".jld2"

    jldsave("rate_matrices/" * filename; rates_tensor)
end