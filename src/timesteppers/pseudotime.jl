export PseudoTimeStepper, solve!

"""Simple time stepper for solving nonlinear, time-independent problems."""
Base.@kwdef mutable struct PseudoTimeStepper <: AbstractTimeStepper
  """current time"""
  t::Float64 = 0.0
  """current step"""
  step::Int = 1
  """time step"""
  Δt::Float64 = 0.01
  """initial time"""
  t0::Float64 = 0.0
  """final time"""
  maxtime::Float64 = 1.0
  """Maximum number of steps"""
  maxsteps::Int = Int((maxtime - t) / Δt + 1)

  """optionally allow DivergedMaxIterations to be considered converged"""
  allow_converged_max_its::Bool = false

  """Solution vector"""
  U::Array{Float64,2} = zeros(0, 0)

  """nonlinear solver"""
  solver::T where {T<:AbstractNonlinearSolver}
  """postprocessing functions"""
  postprocess::Vector{Function} = []
  """output data"""
  output_data::Dict{Symbol,AbstractArray} = Dict()
end

function PseudoTimeStepper(solver; kwargs...)
  ndof = numdof(solver.fem)
  ts = PseudoTimeStepper(; solver=solver, kwargs...)
  ts.U = zeros(Float64, ts.maxsteps, ndof)
  ts
end

"""
    set_residual_fn!(ts::PseudoTimeStepper, residual_fn!::Function)

Set the residual function for dynamic problems. 
The function signature is:

    residual_fn!(fem::FEMModel, U::Vector{Float64}, R::Vector{Float64}, ctx)

  - `fem` is the finite element model
  - `U` is the solution vector
  - `R` is the residual vector to be computed (output)
  - `ctx` is the user-provided context, set through `ctx!`
"""
set_residual_fn!(ts::PseudoTimeStepper, residual_fn!::Function) = set_residual_fn!(ts.solver, residual_fn!)

"""
    set_jacobian_fn!(ts::PseudoTimeStepper, jacobian_fn!::Function)

Set the Jacobian function for dynamic problems.
The function signature is:

    jacobian_fn!(fem::FEMModel, U::Vector{Float64}, K::Matrix{Float64}, ctx)

  - `fem` is the finite element model
  - `U` is the solution vector
  - `K` is the Jacobian matrix to be computed (output)
  - `ctx` is the user-provided context, set through `ctx!`
"""
set_jacobian_fn!(ts::PseudoTimeStepper, jacobian_fn!::Function) = set_jacobian_fn!(ts.solver, jacobian_fn!)

solution(ts::PseudoTimeStepper) = @view ts.U[:, :]
residual(ts::PseudoTimeStepper) = @view ts.residual[:, :]

"""Solve the problem with the given initial guess"""
function solve!(ts::PseudoTimeStepper, U::AbstractVector=zeros(numdof(ts)))
  r = getoutputarray!(ts, :residual)
  num_its = getoutputarray!(ts, :num_its, 1)
  ts.U[step(ts), :] .= U
  ts.t0 = ts.t

  num_its[step(ts)] = 0
  while ts.t < ts.maxtime && step(ts) < ts.maxsteps
    step!(ts, ts.Δt)
    solve!(solver(ts), ts.U[step(ts)-1, :])
    if !is_converged(solver(ts))
      trim!(ts)
      return
    end
    ts.U[step(ts), :] .= solution(solver(ts))
    r[step(ts), :] .= residual(solver(ts))
    num_its[step(ts)] = iterations(solver(ts))
    postprocess!(ts, ts.U[step(ts), :])
  end
end
