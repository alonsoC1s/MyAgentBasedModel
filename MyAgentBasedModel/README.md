# Agent Based Opinion models

## How to run

1. Navigate to the directory either manually or by opening the directory on the
editor.

2. Activate the Julia environment by going into Pkg mode (with the `]` key), and
typing `activate .`. Some editors do this step automatically when properly
configured.

3. Use the code by importing it like any other package:
```
using MyAgentBasedModel

# Create the opinion model over [-2, 2] x [-2, 2]
p = OpinionModelProblem((-2.0, 2.0), (-2.0, 2.0))
```

## Milestones

1. Rewrote the initialization function and made it around 40% faster, and
allocate 70% less memory.

## The Plan

* Come up with reasearch questions

    - Something about the discretization method.

    - Can we find a discretization method that works {consistently, reliably,
    "good"} specifically for the Opinion Dynamics model

* Think about a specific example where the model applies, and tailor the
  parameter search to recreate that scenario.

* Look for open data sets on some phenomenon.

* Decide on the set of parameters for the model we want to study.

* Decide on which discretization method to use, how to estimate transition
  probabilities, etc...