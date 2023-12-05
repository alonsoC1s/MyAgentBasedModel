using MyAgentBasedModel
using JLD2

settings_list = OpinionModelParams[
    OpinionModelParams(), # Standard parameters
    OpinionModelParams(4, 2, 250, 15, 1, 4, 2, 0.5, 0, 0, 10, 100), # Switched agent and media influence strength
    OpinionModelParams(4, 2, 250, 15, 1, 2, 4, 0.5, 0, 0, 5, 100)  # Reduced influencer friction
]

params_desc = [
    "paper_settings",
    "Switched_a_b",
    "small_influencer_friction"
]

for (params, desc) in zip(settings_list, params_desc)
    selectedseed = 420

    # Generate a new opinonmodelproblem
    omp = OpinionModelProblem((-2, 2), (-2, 2); p = params)

    # Solve system, get evolution, compute rate tensor and save
    rX, rY, rZ, rC, rR = solve(omp; seed=selectedseed)

    rates_tensor = time_rate_tensor(rR, rC)

    # Save in JLD2 with the seed as name
    filename = "ratemat_" * string(desc) * ".jld2"

    jldsave("rate_matrices/" * filename; rates_tensor)
end