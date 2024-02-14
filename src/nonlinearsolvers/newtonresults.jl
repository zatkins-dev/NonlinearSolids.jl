export NewtonResult, NewtonFEMResult, trim!, numsteps, maxits, numdof, dim, getoutputarray!

mutable struct NewtonResult <: AbstractSolverResult
  d::Array{Float64}  # maxsteps×dim
  dₙᵏ::Array{Float64,3} # maxsteps×maxits×dim
  res_d::Array{Float64,2} # maxsteps×maxits
  num_its::Array{Int64,1} # maxsteps
  function NewtonResult(dim; numsteps=20, maxits=100)
    d = zeros(numsteps, dim)
    dₙᵏ = zeros(numsteps, maxits, dim)
    res_d = zeros(numsteps, maxits)
    num_its = zeros(Int, numsteps)
    return new(d, dₙᵏ, res_d, num_its)
  end
end

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
  end
end

mutable struct NewtonFEMResult <: AbstractSolverResult
  d::Array{Float64,3}  # maxsteps×(dof×dim)
  dₙᵏ::Array{Float64,4} # maxsteps×maxits×(dof×dim)
  res_d::Array{Float64,2} # maxsteps×maxits
  num_its::Array{Int64,1} # maxsteps
  output_data::Dict{Symbol,Any}
  function NewtonFEMResult(ndof, dim; num_steps=20, maxits=100)
    d = zeros(num_steps, ndof, dim)
    dₙᵏ = zeros(num_steps, maxits, ndof, dim)
    res_d = zeros(num_steps, maxits)
    num_its = zeros(Int, num_steps)
    return new(d, dₙᵏ, res_d, num_its, Dict())
  end
end

numsteps(result::NewtonFEMResult) = size(result.d, 1)
maxits(result::NewtonFEMResult) = size(result.dₙᵏ, 2)
numdof(result::NewtonFEMResult) = size(result.d, 2)
dim(result::NewtonFEMResult) = size(result.d, 3)

function getoutputarray!(result::NewtonFEMResult, key::Symbol, dims=missing)
  if ismissing(dims)
    dims = (numsteps(result), numdof(result),)
  end
  if !haskey(result.output_data, key)
    result.output_data[key] = zeros(Float64, dims)
  end
  return result.output_data[key]
end

function trim!(result::NewtonFEMResult, maxsteps=nothing)
  s = maximum(result.num_its)
  if isnothing(maxsteps)
    result.dₙᵏ = result.dₙᵏ[:, 1:s, :, :]
    result.res_d = result.res_d[:, 1:s]
  else
    result.d = result.d[1:maxsteps, :, :]
    result.dₙᵏ = result.dₙᵏ[1:maxsteps, 1:s, :, :]
    result.res_d = result.res_d[1:maxsteps, 1:s]
    result.num_its = result.num_its[1:maxsteps]
  end
end
