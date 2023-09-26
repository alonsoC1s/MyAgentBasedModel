using JLD2, Plots, Pkg

theme(:ggplot2)

# Pkg.activate("Numerics4/")
prefixname = "fixed"
name = prefixname * ".gif"

function simple_frame(X, C, t, ttl)
    shapes = [:star, :dtriangle]
    colors = [:red, :green, :blue, :black]

    c_idx = findfirst.(C[:, :, t] |> eachrow)

    p = scatter(eachcol(X[:, :, t])...,
        c=colors[c_idx],
        legend=:none,
        xlims=(-2, 2),
        ylims=(-2, 2),
        markerstrokewidth=1.5,
        title = ttl
    )

    return p
end

function comparison_frame(t, X, C, lX, lC)
    p1 = simple_frame(X, C, t, "New")
    p2 = simple_frame(lX, lC, t, "Original")
    plot(p1, p2, plot_title = "t = $(round(0.01 * t; digits=4))")
end

function comparison_evolution(X, C, lX, lC)
    size(X, 3) != size(lX, 3) && throw(ArgumentError("lX & X dont match size"))
    T = size(X, 3)
    anim = @animate for t = 1:T
        comparison_frame(t, X, C, lX, lC)
    end

    return gif(anim, fps=15, "img/" * name)
end

# Getting pre-computed data
new = load("test_data/$(prefixname)_n.jld2")
legacy = load("test_data/$(prefixname)_l.jld2")

nX, nC = new["X"], new["C"]
lX, lC = legacy["X"], legacy["C"]

comparison_evolution(nX, nC, lX, lC)