module MyAgentBasedModel

# Imports...
using LinearAlgebra

# Exports...
export
    _boolean_combinator,
    _orthantize,
    _place_influencers,
    OpinionModelProblem,
    OpinionModelParams,
    AgAg_attraction,
    InfAg_attraction,
    MedAg_attraction,
    solve,
    plot_evolution,
    plot_frame,
    time_rate_tensor,
    _ag_ag_echo_chamber

include("utils.jl")
include("abm.jl")

end # module MyAgentBasedModel
