module NonlinearSolids
using LinearAlgebra
using Revise

export AbstractSolverResult

# Abstract types
abstract type AbstractSolverResult end

# Utility functions
include("utils.jl")

# Results
include("nonlinearsolvers/newtonresults.jl")
include("nonlinearsolvers/arclengthresults.jl")

# FEM
include("fem/element.jl")
include("fem/quadrature.jl")
include("fem/shapefunctions.jl")
include("fem/boundary.jl")
include("fem/fem.jl")

# Nonlinear Solvers
include("nonlinearsolvers/newton.jl")
include("nonlinearsolvers/newtonarclength.jl")
end
