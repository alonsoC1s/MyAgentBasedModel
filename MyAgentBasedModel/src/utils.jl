
using Statistics, Random


## Maybe Fix:
# _boolean_combinator gives the orthant's signatures in a "weird order". Instead
# of numbering counter clokwise, the numbering starts at the all-positive
# quadrant, then moves Down, then Up-Left, then down. In 3 dimensions:
# If z >= 0     If z < 0
#      |            |
#   5 | 1         6 | 2
# ---------     --------- 
#   7 | 3         8 | 4
#     |             |
"""
    _boolean_combinator(n::Integer)::BitMatrix

Given `n` boolean variables, generates all the possible logical combinations for
those variables. e.g for A, B, it generates (true, true), (true, false), (false,
true), (false, false).

Code inspired by TruthTables.jl (macroutils.jl)
https://github.com/eliascarv/TruthTables.jl/blob/main/src/macroutils.jl#L44
"""
function _boolean_combinator(n::Integer)::BitMatrix
    bools = [true, false]
    outers = [2^x for x in 0:n-1]
    inners = reverse(outers)
    return reduce(hcat,
        [repeat(bools; inner, outer) for (inner, outer) in zip(inners, outers)]
    )
end


"""
    _orthantize(X :: AbstractArray{Float64, N})

Classifies the `n` agents in X into the 2^`N` orthants (or quadrants) by
returning an adjacency matrix of size n × 2^`N` where the i-th row has a single
true entry representing which orthant the i-th agent is in.
"""
function _orthantize(X)
    N = size(X, 2) # Dimensions of the problem (i.e N if R^N is the problem space)
    orth_chart = similar(X, Bool)
    # For each row of X (i.e agent), we check if it's on the >=0 semispace for each variable
    for (agent_coord, coord_idx) = zip(eachcol(X), axes(orth_chart, 2))
        orth_chart[:, coord_idx] = agent_coord .>= 0
    end

    # We compare the "orth_chart" with the sequence of booleans that
    # characterize an orthant. e.g The first quadrant (in R^2) can be
    # characterized as (true, true), since x and y are >= 0. The first octant is
    # then (true, true, true) and so on.

    # The "signature" of an orthant is a sequence of booleans s.t the i-th entry
    # is true if the i-th coordinate of points within is >= 0.
    orth_signatures = _boolean_combinator(N)

    orth_class = falses(size(X, 1), 2^N)

    for orthant_id = axes(orth_signatures, 1) # FIXME: Iter over rows, not cols
        signature = orth_signatures[orthant_id, :]
        for agent_id = axes(orth_chart, 1)
            agent = orth_chart[agent_id, :]

            orth_class[agent_id, orthant_id] = all(agent .== signature)
        end
    end

    return orth_class
end

"""
    _place_influencers(X::AbstractArray{Float64, N}, AgInfNet::BitMatrix)

Returns the positions of the influencers in the problem space calculated as the
barycenter of the agents in each orthant
"""
function _place_influencers(X, AgInfNet)
    L = size(AgInfNet, 2) # Number of influencers
    N = size(X, 2) # Dimension of the problem space

    I = similar(X, L, N)
    for (infl_number, orthant_mask) = enumerate(eachcol(AgInfNet)) # FIXME: Use `pairs` instead of enumerate
        orthant_members_idx = findall(orthant_mask)
        I[infl_number, :] = mean(X[orthant_members_idx, :]; dims=1)
    end

    return I
end

"""
    _media_network(n::Int, M::Int)

Returns the adjacency matrix of `n` agents to `M` media outlets such that each
agent is connected to at least one media outlet, possible more.
"""
function _media_network(n, M)
    C = rand([true, false], n, M)
    spots_failing = findall(all(.!c) for c = eachrow(C))
    fix_index = rand(1:M, length(spots_failing))

    for (row, fix_col) = zip(axes(C, 1), fix_index)
        C[row, fix_col] .= true
    end

    return C |> BitMatrix
end