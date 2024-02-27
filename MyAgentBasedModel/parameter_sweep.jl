using Pkg

Pkg.activate(".")

using MyAgentBasedModel
using NPZ, JLD2

##
Threads.@threads for i = 1:10
    Threads.@threads for j = 'a':'b'
        println("$((i, j))")
    end
end