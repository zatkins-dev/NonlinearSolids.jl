module NonlinearSolids
using LinearAlgebra
using Revise

export newtonraphson, modifiednewton, newtonarclength, NewtonResult, ArcLengthResult, trim!

# Utility functions
include("utils.jl")

# Results
include("NewtonResult.jl")
include("ArcLengthResult.jl")

# Nonlinear Solvers
include("newton.jl")
include("newtonarclength.jl")

end
