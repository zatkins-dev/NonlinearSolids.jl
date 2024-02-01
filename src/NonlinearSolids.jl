module NonlinearSolids
using LinearAlgebra
using Revise

export newtonraphson, modifiednewton, newtonarclength, NewtonResult, ArcLengthResult, AbstractSolverResult, trim!

# Abstract types
abstract type AbstractSolverResult end

# Utility functions
include("utils.jl")

# Results
include("NewtonResult.jl")
include("ArcLengthResult.jl")

# Nonlinear Solvers
include("newton.jl")
include("newtonarclength.jl")

end
