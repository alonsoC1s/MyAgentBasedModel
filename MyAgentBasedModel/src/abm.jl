
"""
    OpinionModelParams

Wrapper for the Opinion Dynamics model parameters. The available parameters are:
- `L::Int`: number of influencers
- `M::Int`: number of media outlets
- `n::Int`: number of agents
- `η`: constant for the rate of influencer hopping for agents
- `a`: interaction strength between agents
- `b`: interaction strength between agents and influencers
- `c`: interaction strength between agents and media outlets
- `σ`: noise constant of the agent's stochastic movement
- `̂σ`: (\\hat\\sigma) noise constant for influencer movement
- `̃σ` (\\tilde\\sigma) noise constant for media movement
- `frictionI`: friction constant for influencers
- `frictionM`: friction constant for media outlets
"""
struct OpinionModelParams{T<:Real} # FIXME: Parametrizing on T is unecessary
    L::Int
    M::Int
    n::Int
    η::T
    a::T
    b::T
    c::T
    σ::T
    σ̂::T
    σ̃::T
    frictionI::T
    frictionM::T
end

function OpinionModelParams(L, M, n, η, a, b, c, σ, σ̂, σ̃, FI, FM)
    OpinionModelParams(L, M, n, promote(η, a, b, c, σ, σ̂, σ̃, FI, FM)...)
end

function OpinionModelParams()
    OpinionModelParams(4, 2, 250, 15, 1, 4, 2, 0.5, 0, 0, 10, 100)
end

struct OpinionModelProblem{T<:Real}
    p::OpinionModelParams{T} # Model parameters
    X::AbstractVecOrMat{T} # Array of Agents' positions
    M::AbstractVecOrMat{T} # Array of Media positions
    I::AbstractVecOrMat{T} # Array of Influencers' positions
    AgInfNet::AbstractMatrix{Bool} # Adjacency matrix of Agents-Influencers
    AgAgNet::AbstractMatrix{Bool} # Adjacency matrix of Agent-Agent interactions
    AgMedNet::AbstractMatrix{Bool} # Agent-media correspondence vector
end

function Base.show(io::IO, omp::OpinionModelProblem{T}) where {T}
    print(
        """
        $(size(omp.X, 2))-dimensional Agent Based Opinion Model with:
        - $(omp.p.n) agents
        - $(omp.p.L) influencers
        - $(omp.p.M) media outlets
        """
    )
end

function OpinionModelProblem(dom::Vararg{Tuple{T,T},N};
    p=OpinionModelParams()) where {N,T<:Real}
    # Place agents uniformly distributed across the domain
    X = reduce(hcat, [rand(Uniform(t...), p.n) for t in dom]) # p.n × N matrix

    # We consider just 2 media outlets at the "corners"
    M = vcat(
        fill(-1.0, (1, N)),
        fill(1.0, (1, N))
    ) # FIXME: Parametrize on eltype T instead of hard-coded 1.0

    # We divide the domain into orthants, and each orthant has 1 influencer
    p.L != 2^N && throw(ArgumentError("Number of influencers has to be 2^n"))

    # Create Agent-Influence network (n × L) by grouping individuals into quadrants
    # i,j-th entry is true if i-th agent follows the j-th influencer
    AgInfNet = _orthantize(X)

    # Placing the influencers as the barycenter of agents per orthant
    I = _place_influencers(X, AgInfNet)

    # Every agent interacts with every other agent (including themselves)
    AgAgNet = trues(p.n, p.n)

    # Assign agents to media outlet randomly
    AgMedNet = rand([true, false], p.n, p.M)

    if N == 1
        X = vec(X)
        M = vec(M)
        I = vec(I)
    end

    return OpinionModelProblem(p, X, M, I, AgInfNet, AgAgNet, AgMedNet)
end

"""
    AgAg_attraction(X, AgAgNet)

Calculate the force of attraction on agents exerted by other agents they are
connected to, as determined by `AgAgNet`, the adjacency matrix.
"""
# function AgAg_attraction(X, AgAgNet; φ::Function = x -> exp(-x))
function AgAg_attraction(omp::OpinionModelProblem; φ::Function=x -> exp(-x))
    X, AgAgNet = omp.X, omp.AgAgNet
    force = similar(X)
    for j = axes(force, 1)
        neighboors = findall(AgAgNet[j, :])

        if isempty(neighboors)
            force[j, :] = zeros(eltype(X), 1, 2)
        else
            fi = zeros(eltype(X), 1, 2)
            wsum = zero(eltype(X))
            for neighboor_idx in neighboors
                d = X[neighboor_idx, :] - X[j, :]
                w = φ(norm(d))
                fi = fi + w * d'
                wsum = wsum + w
            end
            force[j, :] = fi ./ wsum
        end
    end
    return force
end

"""
    MedAg_attraction(omp::OpinionModelProblem)

Calculates the Media-Agent attraction force for all agents.
"""
function MedAg_attraction(omp::OpinionModelProblem)
    X, M, B = omp.X, omp.M, omp.AgMedNet
    force = similar(X)
    for i = axes(force, 1)
        force[i, :] = sum(B[i, :]) ./ sum(B[i, m] * (M[m, :] - X[i, :]) for m = axes(B, 2))
    end

    return force
end

"""
    InfAg_attraction(omp::OpinionModelProblem)

Calcultates the Influencer-Aagent attraction force for all agents.
"""
function InfAg_attraction(omp::OpinionModelProblem)
    X, Z, C = omp.X, omp.I, omp.AgMedNet
    force = similar(X)
    for i = axes(force, 1)
        force[i, :] = sum(C[i, :]) ./ sum(C[i, m] * (Z[m, :] - X[i, :]) for m = axes(C, 2))
    end

    return force
end

# Prueba small