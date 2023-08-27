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
        # force[i, :] = sum(C[i, m] * (Z[m, :] - X[i, :]) for m = axes(C, 2)) # ./ sum(C[i, :])
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

function solve(omp::OpinionModelProblem; Nt=100, dt=0.01)
    X, Y, Z, A, B, C = get_values(omp)
    σ, n, Γ, γ, = omp.p.σ, omp.p.n, omp.p.frictionM, omp.p.frictionI
    M, L = omp.p.M, omp.p.L
    d = size(X, 2)
    σ̂, σ̃ = omp.p.σ̂, omp.p.σ̃

    # Allocating solutions & setting initial conditions
    rX = zeros(size(X, 1), size(X, 2), Nt)
    rY = zeros(size(Y, 1), size(Y, 2), Nt)
    rZ = zeros(size(Z, 1), size(Z, 2), Nt)

    rX[:, :, begin] = X
    rY[:, :, begin] = Y
    rZ[:, :, begin] = Z

    # Solve with Euler-Maruyama
    for i = 1:Nt-1
        # X, Y, Z = selectdim.([rX, rY, rZ], 3, i)
        # X_next, Y_next, Z_next = selectdim.([rX, rY, rZ], 3, i + 1)
        # X, X_next = selectdim(rX, 3, i), selectdim(rX, 3, i+1)
        # Y, Y_next = selectdim(rY, 3, i), selectdim(rY, 3, i+1)
        # Z, Z_next = selectdim(rZ, 3, i), selectdim(rZ, 3, i+1)

        X = rX[:, :, i]
        Y = rY[:, :, i]
        Z = rZ[:, :, i]

        # Agents movement
        # FA = agent_drift(X, Y, Z, A, B, C, omp.p)
        # X_next .= X + dt * FA + σ * sqrt(dt) * randn(n, d)

        # # Media movements
        # FM = media_drift(X, Y, B)
        # Y_next .= Y + (dt / Γ) * FM + (σ̃ / Γ) * sqrt(dt) * randn(M, d)

        # # Influencer movements
        # FI = influencer_drift(X, Z, C)
        # Z_next .= Z + (dt / γ) * FI + (σ̂ / γ) * sqrt(dt) * randn(L, d)

        # Agents movement
        FA = agent_drift(X, Y, Z, A, B, C, omp.p)
        rX[:, :, i+1] .= X + dt * FA + σ * sqrt(dt) * randn(n, d)

        # Media movements
        FM = media_drift(X, Y, B)
        rY[:, :, i+1] .= Y + (dt / Γ) * FM + (σ̃ / Γ) * sqrt(dt) * randn(M, d)

        # Influencer movements
        FI = influencer_drift(X, Z, C)
        rZ[:, :, i+1] .= Z + (dt / γ) * FI + (σ̂ / γ) * sqrt(dt) * randn(L, d)

    end

    return rX
end