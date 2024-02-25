module NonlinearSolids
using LinearAlgebra
using Revise
using Format

export AbstractSolverResult

# Abstract types
abstract type AbstractSolverResult end
abstract type AbstractNonlinearSolver end
abstract type AbstractTimestepper end

# Utility functions
include("utils.jl")
include("nonlinearsolvers/types.jl")
include("timesteppers/types.jl")

# FEM
include("fem/element.jl")
include("fem/quadrature.jl")
include("fem/shapefunctions.jl")
include("fem/boundary.jl")
include("fem/fem.jl")

# Time Steppers
include("timesteppers/pseudotime.jl")
include("timesteppers/newmark.jl")

# Nonlinear Solvers
include("nonlinearsolvers/newton.jl")
include("nonlinearsolvers/newtonarclength.jl")

# Results
include("nonlinearsolvers/newtonresults.jl")
include("nonlinearsolvers/arclengthresults.jl")
end
