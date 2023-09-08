using LinearAlgebra
using Distributions

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
    AgInfNet::BitMatrix # Adjacency matrix of Agents-Influencers
    AgAgNet::BitMatrix # Adjacency matrix of Agent-Agent interactions
    AgMedNet::BitMatrix # Agent-media correspondence vector
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
    AgInfNet = _orthantize(X) |> BitMatrix

    # Placing the influencers as the barycenter of agents per orthant
    I = _place_influencers(X, AgInfNet)

    # Every agent interacts with every other agent (including themselves)
    AgAgNet = trues(p.n, p.n) |> BitMatrix

    # Assign agents to media outlet randomly
    # FIXME: Agents should be connected to at least 1 media outlet
    AgMedNet = _media_network(p.n, p.M)

    if N == 1
        X = vec(X)
        M = vec(M)
        I = vec(I)
    end

    return OpinionModelProblem(p, X, M, I, AgInfNet, AgAgNet, AgMedNet)
end

function get_values(omp::OpinionModelProblem)
    return omp.X, omp.M, omp.I, omp.AgAgNet, omp.AgMedNet, omp.AgInfNet
end

function AgAg_attraction(X::AbstractVecOrMat{T}, A::BitMatrix; φ=x -> exp(-x)) where {T}
    force = similar(X)
    for j = axes(force, 1)
        neighboors = findall(A[j, :])

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
    AgAg_attraction(X, AgAgNet)

Calculate the force of attraction on agents exerted by other agents they are
connected to, as determined by `AgAgNet`, the adjacency matrix.
"""
# function AgAg_attraction(X, AgAgNet; φ::Function = x -> exp(-x))
function AgAg_attraction(omp::OpinionModelProblem{T}; φ=x -> exp(-x)) where {T}
    X, A = omp.X, omp.AgAgNet
    return AgAg_attraction(X, A)
end

function MedAg_attraction(X::T, M::T, B::BitMatrix) where {T<:AbstractVecOrMat}
    force = similar(X)
    # FIXME: Can be written even more compactly

    # Detect early if an agent is not connected to any Media Outlets
    if !(any(B; dims=2) |> all)
        throw(ErrorException("Model violation detected: An agent is disconnected " *
                             "from all media outlets."))
    end

    for i = axes(X, 1)
        media_idx = findfirst(B[i, :])
        force[i, :] = M[media_idx, :] - X[i, :]
    end

    return force
end

"""
    MedAg_attraction(omp::OpinionModelProblem)

Calculates the Media-Agent attraction force for all agents.
"""
function MedAg_attraction(omp::OpinionModelProblem)
    return MedAg_attraction(omp.X, omp.M, omp.AgMedNet)
end

function InfAg_attraction(X::T, Z::T, C::BitMatrix) where {T<:AbstractVecOrMat}
    force = similar(X)

    # Detect early if an agent doesn't follow any influencers
    if !(any(C; dims=2) |> all)
        throw(ErrorException("Model violation detected: An Agent doesn't follow " *
                             "any influencers"))
    end

    for i = axes(X, 1)
        # force[i, :] = sum(C[i, m] * (Z[m, :] - X[i, :]) for m = axes(C, 2)) # ./ count(C[i, :])
        influencer_idx = findfirst(C[i, :])
        force[i, :] = Z[influencer_idx, :] - X[i, :]
    end

    return force
end

"""
    InfAg_attraction(omp::OpinionModelProblem)

Calcultates the Influencer-Agent attraction force for all agents.
"""
function InfAg_attraction(omp::OpinionModelProblem)
    X, Z, C = omp.X, omp.I, omp.AgInfNet
    return InfAg_attraction(X, Z, C)
end

# function agent_force!(du, u, p, t)
#     a, b, c = p.p.a, p.p.b, p.p.c
#     A, B, C = p.AgAgNet, p.AgInfNet, p.AgMedNet
#     M, Z = p.M, p.I
#     du = a * AgAg_attraction(u, A) + b * MedAg_attraction(u, M, B) + c * InfAg_attraction(u, Z, C)
# end

function follower_average(X::AbstractVecOrMat, Network::BitMatrix)
    mass_centers = zeros(size(Network, 2), size(X, 2))

    # Detect early if one outlet/influencer has lost all followers i.e a some column is empty
    lonely_outlets = Int[]
    if !(any(Network; dims=1) |> all)
        # Exclude this index of the calculations and set zeros manually to the results
        v = any(Network; dims=1) |> (collect ∘ vec) # Hack to force v into a Vector{bool}
        append!(lonely_outlets, findall(!, v))
    end

    # Calculate centers of mass, excluding the outlets left alone to avoid div by zero
    for m = setdiff(axes(Network, 2), lonely_outlets)
        # Get the index of all the followers of m-th medium
        ms_followers = Network[:, m] |> findall
        # Store the col-wise average for the subset of X that contains the followers
        mass_centers[m, :] = mean(X[ms_followers, :]; dims=1)
    end

    return mass_centers
end

function agent_drift(X::T, M::T, I::T, A::Bm, B::Bm, C::Bm,
    p::OpinionModelParams) where {T<:AbstractVecOrMat,Bm<:BitMatrix}
    a, b, c = p.a, p.b, p.c
    return a * AgAg_attraction(X, A) + b * MedAg_attraction(X, M, B) +
           c * InfAg_attraction(X, I, C)
end

function media_drift(X::T, Y::T, B::Bm; f=identity) where {T<:AbstractVecOrMat,
    Bm<:BitMatrix}
    force = similar(Y)
    x_tilde = follower_average(X, B)
    force = f.(x_tilde .- Y)

    return force
end

function influencer_drift(X::T, Z::T, C::Bm; g=identity) where {T<:AbstractVecOrMat,
    Bm<:BitMatrix}
    force = similar(Z)
    x_hat = follower_average(X, C)
    force = g.(x_hat .- Z)

    return force
end

"""
    followership_ratings(B, C)

Calculates the rate of individuals that follow both the `m`-th media outlet and
the `l`-th influencer. The `m,l`-th entry of the output corresponds to the
proportion of agents that follows `m` and `l`.

# Arguments:
- `B::BitMatrix`: Adjacency matrix of the agents to the media outlets
- `C::BitMatrix`: Adjacency matrix of the agents to the influencers
"""
function followership_ratings(B::BitMatrix, C::BitMatrix)
    n, M = size(B)
    L = size(C, 2)

    R = zeros(M, L)
    for m = 1:M
        audience_m = findall(B[:, m])
        R[m, :] = count(C[audience_m, :]; dims=1) ./ n
    end

    return R
end


function influencer_switch_rates(X::T, Z::T, B::Bm, C::Bm, η;
    ψ=x -> exp(-x), r=relu) where {T<:AbstractVecOrMat,Bm<:BitMatrix}

    # Compute the followership rate for media and influencers
    rate_m_l = followership_ratings(B, C)
    # attractiveness is the total proportion of followers per influencer
    # Divide each col of rates over sum(n_(m, l) for m = 1:M)
    # FIXME: Rename as structural similarity
    struct_similarity = rate_m_l ./ sum(rate_m_l; dims=1)

    # Computing distances of each individual to the influencers
    D = zeros(size(X, 1), size(Z, 1))
    for (i, agent_i) = pairs(eachrow(X))
        for (l, influencer_l) = pairs(eachrow(Z))
            D[i, l] = ψ(norm(agent_i - influencer_l))
        end
    end

    # Calculating switching rate based on eq. (6)
    R = zeros(size(X, 1), size(Z, 1))
    for (j, agentj_media) = pairs(eachrow(B))
        m = findfirst(agentj_media)
        R[j, :] = η * D[j, :] .* r.(struct_similarity[m, :])
    end

    return R
end

function switch_influencer(C::Bm, X::T, Z::T, B::Bm, η) where {Bm<:BitMatrix,
    T<:AbstractVecOrMat}
    rates = influencer_switch_rates(X, Z, B, C, η)
    RC = similar(C)
    L = size(Z, 1)

    choices = [I(L)[:, i] |> BitVector for i = 1:L]

    for j = axes(X, 1)
        dc = DiscreteNonParametric(1:L, rates[j, :] ./ sum(rates[j, :]))
        new_influencer_idx = rand(dc)
        new_influencer_idx != findfirst(C[j, :]) && @info "agent $(j) switched from influencer $(findfirst(C[j, :])) to $(new_influencer_idx)"
        RC[j, :] = choices[new_influencer_idx]
    end

    return RC
end

# TODO: test

function solve(omp::OpinionModelProblem{T}; Nt=100, dt=0.01) where {T}
    X, Y, Z, A, B, C = get_values(omp)
    σ, n, Γ, γ, = omp.p.σ, omp.p.n, omp.p.frictionM, omp.p.frictionI
    M, L = omp.p.M, omp.p.L
    d = size(X, 2)
    σ̂, σ̃ = omp.p.σ̂, omp.p.σ̃
    η = omp.p.η

    # Allocating solutions & setting initial conditions
    rX = zeros(T, n, d, Nt)
    rY = zeros(T, M, d, Nt)
    rZ = zeros(T, L, d, Nt)
    rC = zeros(Bool, n, L, Nt)

    rX[:, :, begin] = X
    rY[:, :, begin] = Y
    rZ[:, :, begin] = Z
    rC[:, :, begin] = C

    # Solve with Euler-Maruyama
    for i = 1:Nt-1
        X = view(rX, :, :, i)
        Y = view(rY, :, :, i)
        Z = view(rZ, :, :, i)

        # Agents movement
        FA = agent_drift(X, Y, Z, A, B, C, omp.p)
        rX[:, :, i+1] .= X + dt * FA + σ * sqrt(dt) * randn(n, d)

        # Media movements
        FM = media_drift(X, Y, B)
        rY[:, :, i+1] .= Y + (dt / Γ) * FM + (σ̃ / Γ) * sqrt(dt) * randn(M, d)

        # Influencer movements
        FI = influencer_drift(X, Z, C)
        rZ[:, :, i+1] .= Z + (dt / γ) * FI + (σ̂ / γ) * sqrt(dt) * randn(L, d)

        # Change influencers
        view(rC, :, :, i+1) .= switch_influencer(C, X, Z, B, η)

    end

    return rX, rY, rZ, rC
end