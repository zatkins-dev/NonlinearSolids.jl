export ArcLengthResult, trim!

mutable struct ArcLengthResult <: AbstractSolverResult
  d::Array{Float64}  # maxsteps×dim
  dₙᵏ::Array{Float64,3} # maxsteps×maxits×dim
  λ::Array{Float64,1}  # maxsteps×maxits
  λₙᵏ::Array{Float64,2} # maxsteps×maxits
  res_d::Array{Float64,2} # maxsteps×maxits
  res_λ::Array{Float64,2} # maxsteps×maxits
  num_its::Array{Int64,1} # maxsteps
  num_steps::Int64
  function ArcLengthResult(dim; maxsteps=20, maxits=100)
    d = zeros(maxsteps, dim)
    dₙᵏ = zeros(maxsteps, maxits, dim)
    λ = zeros(maxsteps)
    λₙᵏ = zeros(maxsteps, maxits)
    res_d = zeros(maxsteps, maxits)
    res_λ = zeros(maxsteps, maxits)
    num_its = zeros(Int, maxsteps)
    return new(d, dₙᵏ, λ, λₙᵏ, res_d, res_λ, num_its, 0)
  end
end


function trim!(result::ArcLengthResult)
  maxits = maximum(result.num_its)
  result.d = result.d[1:result.num_steps, :]
  result.λ = result.λ[1:result.num_steps]
  result.dₙᵏ = result.dₙᵏ[1:result.num_steps, 1:maxits, :]
  result.λₙᵏ = result.λₙᵏ[1:result.num_steps, 1:maxits]
  result.res_λ = result.res_λ[1:result.num_steps, 1:maxits]
  result.res_d = result.res_d[1:result.num_steps, 1:maxits]
  result.num_its = result.num_its[1:result.num_steps]
end
