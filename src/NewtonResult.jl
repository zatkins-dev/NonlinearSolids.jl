mutable struct NewtonResult <: AbstractSolverResult
  d::Array{Float64}  # maxsteps×dim
  dₙᵏ::Array{Float64,3} # maxsteps×maxits×dim
  res_d::Array{Float64,2} # maxsteps×maxits
  num_its::Array{Int64,1} # maxsteps
end

function NewtonResult(dim; numsteps=20, maxits=100)
  d = zeros(numsteps, dim)
  dₙᵏ = zeros(numsteps, maxits, dim)
  res_d = zeros(numsteps, maxits)
  num_its = zeros(Int, numsteps)
  return NewtonResult(d, dₙᵏ, res_d, num_its)
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
