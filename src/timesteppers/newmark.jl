using LinearAlgebra

export NewmarkTimeStepper, set_residual_fn!, set_jacobian_fn!, update_predictors!, update_state!, velocity, acceleration, solution, residual, solve!, ∂U̇̇∂U

Base.@kwdef mutable struct NewmarkTimeStepper <: AbstractTimeStepper
  """Displacement integration parameter"""
  β::Float64 = 0.25
  """Velocity integration parameter"""
  γ::Float64 = 0.5
  """Current time"""
  t::Float64 = 0.0
  """Current step"""
  step::Int = 1
  """Time step"""
  Δt::Float64 = 0.01
  """Initial time"""
  t0::Float64 = 0.0
  """Final time"""
  maxtime::Float64 = 1.0
  """Maximum number of steps"""
  maxsteps::Int = Int((maxtime - t) / Δt + 1)
  """Displacement predictor"""
  Ũ::Vector{Float64} = zeros(0)
  """Velocity predictor"""
  Ũ̇::Vector{Float64} = zeros(0)
  """Solution vector"""
  U::Array{Float64,2} = zeros(0, 0)
  """Velocity vector"""
  U̇::Array{Float64,2} = zeros(0, 0)
  """Acceleration vector"""
  U̇̇::Array{Float64,2} = zeros(0, 0)

  """nonlinear solver"""
  solver
  """postprocessing functions"""
  postprocess::Vector{Function} = []
  """output data"""
  output_data::Dict{Symbol,AbstractArray} = Dict()
end

function NewmarkTimeStepper(solver; kwargs...)
  ndof = numdof(solver.fem)
  ts = NewmarkTimeStepper(; solver=solver, kwargs...)
  ts.Ũ = zeros(Float64, ndof)
  ts.Ũ̇ = zeros(Float64, ndof)
  ts.U = zeros(Float64, ts.maxsteps, ndof)
  ts.U̇ = zeros(Float64, ts.maxsteps, ndof)
  ts.U̇̇ = zeros(Float64, ts.maxsteps, ndof)
  ts
end

"""Get derivative of displacement with respect to acceleration"""
function ∂U̇̇∂U(ts::NewmarkTimeStepper)
  return ts.Δt^2 * ts.β
end

"""
    set_residual_fn!(ts::NewmarkTimeStepper, residual_fn!::Function)

Set the residual function for dynamic problems. 
The function signature is:

    residual_fn!(fem::FEMModel, U::Vector{Float64}, U̇::Vector{Float64}, 
                 U̇̇::Vector{Float64}, R::Vector{Float64}, ctx)

  - `fem` is the finite element model
  - `U` is the solution vector
  - `U̇` is the velocity vector
  - `U̇̇` is the acceleration vector
  - `R` is the residual vector to be computed (output)
  - `ctx` is the user-provided context, set through `ctx!`
"""
function set_residual_fn!(ts::NewmarkTimeStepper, residual_fn!::Function)
  nr_residual = (fem, U̇̇, R, ctx) -> begin
    update_state!(ts, U̇̇)
    applydirichletboundaries!(fem, ts.U[step(ts), :])
    residual_fn!(fem, ts.U[step(ts), :], ts.U̇[step(ts), :], U̇̇, R, ctx)
  end
  set_residual_fn!(solver(ts), nr_residual)
end

"""
    set_jacobian_fn!(ts::NewmarkTimeStepper, jacobian_fn!::Function)

Set the Jacobian function for dynamic problems.
The function signature is:

    jacobian_fn!(fem::FEMModel, U::Vector{Float64}, U̇::Vector{Float64}, 
                 U̇̇::Vector{Float64}, K::Matrix{Float64}, ctx)

  - `fem` is the finite element model
  - `U` is the solution vector
  - `U̇` is the velocity vector
  - `U̇̇` is the acceleration vector
  - `K` is the Jacobian matrix to be computed (output)
  - `ctx` is the user-provided context, set through `ctx!`
"""
function set_jacobian_fn!(ts::NewmarkTimeStepper, jacobian_fn!::Function)
  nr_jacobian = (fem, U̇̇, K, ctx) -> begin
    update_state!(ts, U̇̇)
    applydirichletboundaries!(fem, ts.U[step(ts), :])
    jacobian_fn!(fem, ts.U[step(ts), :], ts.U̇[step(ts), :], U̇̇, K, ctx)
  end
  set_jacobian_fn!(solver(ts), nr_jacobian)
end

function update_predictors!(ts::NewmarkTimeStepper, n::Int=step(ts))
  ts.Ũ .= ts.U[n, :] + ts.Δt * ts.U̇[n, :] + ts.Δt^2 * (0.5 - ts.β) * ts.U̇̇[n, :]
  ts.Ũ̇ .= ts.U̇[n, :] + ts.Δt * (1 - ts.γ) * ts.U̇̇[n, :]
end

function update_state!(ts::NewmarkTimeStepper, U̇̇::AbstractVector{Float64}, n::Int=step(ts))
  ts.U[n, :] .= ts.Ũ + ts.Δt^2 * ts.β * U̇̇
  ts.U̇[n, :] .= ts.Ũ̇ + ts.Δt * ts.γ * U̇̇
end

velocity(ts::NewmarkTimeStepper, range=:) = @view ts.U̇[range, :]
acceleration(ts::NewmarkTimeStepper, range=:) = @view ts.U̇̇[range, :]
solution(ts::NewmarkTimeStepper, range=:) = @view ts.U[range, :]
residual(ts::NewmarkTimeStepper, range=:) = @view ts.residual[range, :]

function solve!(ts::NewmarkTimeStepper, U0=zeros(numdof(ts)), U̇0=zeros(numdof(ts)), U̇̇0=zeros(numdof(ts)))
  r = getoutputarray!(ts, :residual)
  num_its = getoutputarray!(ts, :num_its, 1)
  r[step(ts), :] .= 0.0
  num_its[step(ts)] = 0
  ts.U[step(ts), :] .= U0
  ts.U̇[step(ts), :] .= U̇0
  ts.U̇̇[step(ts), :] .= U̇̇0
  ts.t0 = ts.t
  # TODO: Initialize a0 as M⋅a0 = Fext(t=0) - Fint(U0, t=0)
  while ts.t < ts.maxtime && step(ts) < ts.maxsteps
    update_predictors!(ts, step(ts))
    step!(ts, ts.Δt)
    solve!(ts.solver, ts.U̇̇[step(ts)-1, :])
    if !is_converged(ts.solver)
      trim!(ts)
      return
    end
    ts.U̇̇[step(ts), :] .= solution(ts.solver)
    r[step(ts), :] .= residual(ts.solver)
    num_its[step(ts)] = iterations(ts.solver)
    update_state!(ts, ts.U̇̇[step(ts), :])
    postprocess!(ts, ts.U[step(ts), :])
  end
end
