export ArcLengthResult, trim!

"""Result for an arc length solver"""
mutable struct ArcLengthResult <: AbstractSolverResult
  d::Array{Float64}  # maxsteps×dim
  dₙᵏ::Array{Float64,3} # maxsteps×maxits×dim
  λ::Array{Float64,1}  # maxsteps×maxits
  λₙᵏ::Array{Float64,2} # maxsteps×maxits
  res_d::Array{Float64,2} # maxsteps×maxits
  res_λ::Array{Float64,2} # maxsteps×maxits
  num_its::Array{Int64,1} # maxsteps
  num_steps::Int64
  dim::Int64
  output_data::Dict{Symbol,Any}
  """Create a result for a given dimension"""
  function ArcLengthResult(dim::Int; maxsteps=20, maxits=100)
    d = zeros(maxsteps, dim)
    dₙᵏ = zeros(maxsteps, maxits, dim)
    λ = zeros(maxsteps)
    λₙᵏ = zeros(maxsteps, maxits)
    res_d = zeros(maxsteps, maxits)
    res_λ = zeros(maxsteps, maxits)
    num_its = zeros(Int, maxsteps)
    return new(d, dₙᵏ, λ, λₙᵏ, res_d, res_λ, num_its, 0, dim, Dict())
  end
  """Create a result for a given FEMModel
  
  The solution vector will have `dim(fem) * numdof(fem)` components
  """
  function ArcLengthResult(fem::FEMModel; maxsteps=20, maxits=100)
    d = zeros(maxsteps, dim(fem) * numdof(fem))
    dₙᵏ = zeros(maxsteps, maxits, dim(fem) * numdof(fem))
    λ = zeros(maxsteps)
    λₙᵏ = zeros(maxsteps, maxits)
    res_d = zeros(maxsteps, maxits)
    res_λ = zeros(maxsteps, maxits)
    num_its = zeros(Int, maxsteps)
    return new(d, dₙᵏ, λ, λₙᵏ, res_d, res_λ, num_its, 0, dim(fem), Dict())
  end
end

"""Get the number of steps in the result"""
numsteps(result::ArcLengthResult) = size(result.d, 1)
"""Get the maximum number of iterations per step in the result"""
maxits(result::ArcLengthResult) = size(result.dₙᵏ, 2)
"""Get the spatial dimension of the result"""
dim(result::ArcLengthResult) = result.dim
"""Get the number of degrees of freedom in the result"""
numdof(result::ArcLengthResult) = size(result.d, 2) ÷ dim(result)

"""Get or create an array of size `(numsteps, dims...)` to store output data"""
function getoutputarray!(result::ArcLengthResult, key::Symbol, dims=missing)
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

"""Trim the result to the number of steps and maximum number of iterations per timestep"""
function trim!(result::ArcLengthResult)
  maxits = maximum(result.num_its)
  result.d = result.d[1:result.num_steps, :]
  result.λ = result.λ[1:result.num_steps]
  result.dₙᵏ = result.dₙᵏ[1:result.num_steps, 1:maxits, :]
  result.λₙᵏ = result.λₙᵏ[1:result.num_steps, 1:maxits]
  result.res_λ = result.res_λ[1:result.num_steps, 1:maxits]
  result.res_d = result.res_d[1:result.num_steps, 1:maxits]
  result.num_its = result.num_its[1:result.num_steps]
  for (key, val) in result.output_data
    result.output_data[key] = val[1:result.num_steps, :]
  end
end
