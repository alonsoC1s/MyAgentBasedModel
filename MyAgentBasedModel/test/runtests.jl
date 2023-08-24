module TestUtils

using MyAgentBasedModel, Test
include("../../Numerics4/src/abm.jl")
import
    MyAgentBasedModel._orthantize,
    MyAgentBasedModel._place_influencers

@testset "Tests for auxiliary functions" begin
    ## 1-d test
    ## Agent orthontization
    X = [-1, 1]
    S_expected = [false true; true false]
    S = _orthantize(X)

    @test S == S_expected

    ## Correct influencer placement
    I_expected = [-1; 1]
    I = _place_influencers(X, S)

    # @test I == I_expected

    ## 2-d test
    ## 1 agent per quadrant in ascending clockwise numbering
    X = [
        1 1;
        -1 1;
        -1 -1;
        1 -1
    ]

    S_expected = [
        # Follows the numbering sequence 1-3-4-2 (clockwise)
        true false false false;
        false false true false;
        false false false true;
        false true false false
    ]
    
    S = _orthantize(X)

    @test S == S_expected

    ## Correct influencer placing
    I_expected = [
        1 1;
        1 -1;
        -1 1
        -1 -1;
    ]

    I = _place_influencers(X, S)

    @test I == I_expected

    # 3-d test
    # 1 agent per octant to reproduce sequence 1:8
    X = [
        1 1 1;
        1 1 -1;
        1 -1 1;
        1 -1 -1;
        -1 1 1;
        -1 1 -1;
        -1 -1 1;
        -1 -1 -1;
    ]

    S_expected = Bool[
        1 0 0 0 0 0 0 0;
        0 1 0 0 0 0 0 0;
        0 0 1 0 0 0 0 0;
        0 0 0 1 0 0 0 0;
        0 0 0 0 1 0 0 0;
        0 0 0 0 0 1 0 0;
        0 0 0 0 0 0 1 0;
        0 0 0 0 0 0 0 1;
    ]

    S = _orthantize(X)

    @test S == S_expected

    ## Correct placement of influencers
    I_expected = [
        1 1 1;
        1 1 -1;
        1 -1 1;
        1 -1 -1;
        -1 1 1;
        -1 1 -1;
        -1 -1 1;
        -1 -1 -1;
    ]

    I = _place_influencers(X, S)

    @test I == I_expected
    # TODO: Add one more randomized test
end

@testset "Main functionality" begin
   o = OpinionModelProblem((-2, 2), (-2, 2))

   @test attraction(o.X, o.AgAgNet) == AgAg_attraction(o.X, o.AgAgNet)
end
end