export NewtonResult, NewtonFEMResult, trim!, numsteps, maxits, numdof, dim, getoutputarray!

"""Result for a Newton solver"""
mutable struct NewtonResult <: AbstractSolverResult
  d::Array{Float64}  # maxsteps×dim
  dₙᵏ::Array{Float64,3} # maxsteps×maxits×dim
  res_d::Array{Float64,2} # maxsteps×maxits
  num_its::Array{Int64,1} # maxsteps
  dim::Int
  output_data::Dict{Symbol,Any}
  """Create a result for a given dimension"""
  function NewtonResult(dim::Int; numsteps=20, maxits=100)
    d = zeros(numsteps, dim)
    dₙᵏ = zeros(numsteps, maxits, dim)
    res_d = zeros(numsteps, maxits)
    num_its = zeros(Int, numsteps)
    return new(d, dₙᵏ, res_d, num_its, dim, Dict())
  end
  """Create a result for a given FEMModel

  The solution vector will have `dim(fem) * numdof(fem)` components
  """
  function NewtonResult(fem::FEMModel; numsteps=20, maxits=100)
    d = zeros(numsteps, dim(fem) * numdof(fem))
    dₙᵏ = zeros(numsteps, maxits, dim(fem) * numdof(fem))
    res_d = zeros(numsteps, maxits)
    num_its = zeros(Int, numsteps)
    return new(d, dₙᵏ, res_d, num_its, dim(fem), Dict())
  end
end

"""Get the number of time steps in the result"""
numsteps(result::NewtonResult) = size(result.d, 1)
"""Get the maximum number of iterations per timestep in the result"""
maxits(result::NewtonResult) = size(result.dₙᵏ, 2)
"""Get the spatial dimension of the result"""
dim(result::NewtonResult) = result.dim
"""Get the number of degrees of freedom in the result"""
numdof(result::NewtonResult) = size(result.d, 2) ÷ result.dim

"""Get or create an array of size `(numsteps, dims...)` to store output data"""
function getoutputarray!(result::NewtonResult, key::Symbol, dims=missing)
  if ismissing(dims)
    dims = (numsteps(result), numdof(result),)
  elseif Base.haslength(dims)
    dims = (numsteps(result), dims...)
  else
    dims = (numsteps(result), dims)
  end
  if !haskey(result.output_data, key)
    result.output_data[key] = zeros(Float64, dims)
  end
  return result.output_data[key]
end

"""Trim the result to the maximum number of iterations per timestep"""
function trim!(result::NewtonResult, maxsteps=nothing)
  s = maximum(result.num_its)
  if isnothing(maxsteps)
    result.dₙᵏ = result.dₙᵏ[:, 1:s, :]
    result.res_d = result.res_d[:, 1:s]
  else
    result.d = result.d[1:maxsteps, :]
    result.dₙᵏ = result.dₙᵏ[1:maxsteps, 1:s, :]
    result.res_d = result.res_d[1:maxsteps, 1:s]
    result.num_its = result.num_its[1:maxsteps]
    for (key, val) in result.output_data
      result.output_data[key] = val[1:maxsteps, :]
    end
  end
end
