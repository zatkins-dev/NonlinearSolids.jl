export AbstractNonlinearSolver, NewtonSolver, set_ctx!, set_jacobian_fn!, set_residual_fn!, solution, residual, is_converged, solve!, newton

using Logging

Base.@kwdef mutable struct NewtonSolver <: AbstractNonlinearSolver
  """Type of Newton's method to use, either :standard or :modified"""
  type::Symbol = :standard
  """Absolute tolerance of residual"""
  atol::Float64 = 1e-16
  """Relative tolerance of residual"""
  rtol::Float64 = 1e-16
  """Divergence tolerance of Newton step"""
  dtol::Float64 = 1e6
  """Maximum number of Newton iterations"""
  maxits::Int = 100

  """Optional context for residual and Jacobian functions"""
  ctx::Any = nothing
  """Finite element model"""
  fem::FEMModel
  """Residual function, set by `set_residual_fn!`"""
  residual_fn!::Function = (fem::FEMModel, U::AbstractVector, R::AbstractVector, ctx) -> nothing
  """Jacobian function, set by `set_jacobian_fn!`"""
  jacobian_fn!::Function = (fem::FEMModel, U::AbstractVector, K::AbstractMatrix, ctx) -> nothing

  """Optionally log residual norm"""
  monitor_residual_norm::Bool = false
  """Optionally log reason for convergence"""
  monitor_converged_reason::Bool = false

  """Solution vector"""
  U::Vector{Float64}
  """Step vector"""
  δU::Vector{Float64}
  """Residual vector"""
  R::Vector{Float64}
  """Jacobian matrix"""
  K::Matrix{Float64}

  """Converged/diverged reason"""
  converged_reason::AbstractConvergedReason = REASON_NOT_YET_CONVERGED
  """Number of iterations"""
  iterations::Int = 0
end

"""Create a new Newton solver"""
function NewtonSolver(fem; kwargs...)
  ndof = numdof(fem)
  NewtonSolver(; fem=fem, U=zeros(ndof), δU=zeros(ndof), R=zeros(ndof), K=zeros(ndof, ndof), kwargs...)
end

Base.show(io::IO, solver::NewtonSolver) = println(
  io,
  """
NewtonSolver: $(solver.type)
  Solver options:
    atol: $(solver.atol)
    rtol: $(solver.rtol)
    dtol: $(solver.dtol)
    maxits: $(solver.maxits)
    monitor_residual_norm: $(solver.monitor_residual_norm)
    monitor_converged_reason: $(solver.monitor_converged_reason)
  User-provided functions:
    residual_fn!: $(solver.residual_fn!)
    jacobian_fn!: $(solver.jacobian_fn!)
    fem: $(solver.fem)
"""
)

"""Get solution vector from Newton solver"""
solution(solver::NewtonSolver) = solver.U

"""Get residual vector from Newton solver"""
residual(solver::NewtonSolver) = solver.R

"""Get number of iterations from Newton solver"""
iterations(solver::NewtonSolver) = solver.iterations

"""Get finite element model from Newton solver"""
model(solver::NewtonSolver) = solver.fem

"""Set context for user-provided functions"""
set_ctx!(solver::NewtonSolver, ctx::Any) = solver.ctx = ctx

"""
    set_residual_fn!(solver::NewtonSolver, residual_fn!::Function)

Set function to compute residual vector for Newton's method.
Function signature:

    residual_fn!(fem::FEMModel, U::Vector{Float64}, R::Vector{Float64}, ctx)

    - `fem` is the finite element model
    - `U` is the solution vector
    - `R` is the residual vector to be computed (output)
    - `ctx` is the user-provided context, set through `ctx!`
"""
set_residual_fn!(solver::NewtonSolver, residual_fn!::Function) = solver.residual_fn! = residual_fn!

"""
    set_jacobian_fn!(solver::NewtonSolver, jacobian_fn!::Function)

Set function to compute Jacobian matrix for Newton's method. 
Function signature:

    jacobian_fn!(fem::FEMModel, U::Vector{Float64}, K::Matrix{Float64}, ctx)

  - `ctx` is the user-provided context, set through `ctx!`
  - `U` is the solution vector
  - `K` is the Jacobian matrix to be computed (output)
"""
set_jacobian_fn!(solver::NewtonSolver, jacobian_fn!::Function) = solver.jacobian_fn! = jacobian_fn!

"""
    is_converged(solver::NewtonSolver; allow_maxits=false)

Check if the Newton solver has converged. If `allow_maxits` is `true`, 
then the solver is considered converged if it has reached the maximum number of iterations.
"""
is_converged(solver::NewtonSolver; allow_maxits=false) = is_converged(solver.converged_reason, allow_maxits=allow_maxits)

numdof(solver::NewtonSolver) = numdof(solver.fem)

"""
    solve!(solver::NewtonSolver, U0::Vector{Float64}=zeros(numdof(solver.fem)))

Use Newton's method to solve a nonlinear finite element problem.
"""
function solve!(solver::NewtonSolver, U0::AbstractVector=zeros(numdof(solver.fem)))
  ll_resid = solver.monitor_residual_norm ? Info : Debug
  ll_conv = solver.monitor_converged_reason ? Info : Debug
  solver.U .= U0
  @views begin
    solver.residual_fn!(solver.fem, solver.U[:], solver.R[:], solver.ctx)
    solver.jacobian_fn!(solver.fem, solver.U[:], solver.K[:, :], solver.ctx)
  end
  r = dofs(solver.fem, solver.R)
  norm_r0 = norm(r)
  @logmsg ll_resid "  Initial residual norm: $(norm(r))"
  norm_r0 = norm(r) > eps() ? norm(r) : 1
  norm_r = norm_r0
  solver.iterations = 0 # Initial iteration count
  while solver.iterations < solver.maxits
    if norm_r / norm_r0 < solver.rtol
      solver.converged_reason = REASON_CONVERGED_RELATIVE
      @logmsg ll_conv solver.converged_reason iterations = solver.iterations
      break
    end
    if norm_r < solver.atol
      solver.converged_reason = REASON_CONVERGED_ABSOLUTE
      @logmsg ll_conv solver.converged_reason iterations = solver.iterations
      break
    end
    if solver.type == :standard
      @views begin
        solver.jacobian_fn!(solver.fem, solver.U[:], solver.K[:, :], solver.ctx)
      end
    end
    try
      solver.δU .= expand(solver.fem, -dofs(solver.fem, solver.K) \ dofs(solver.fem, solver.R))
    catch e
      if e isa SingularException
        solver.converged_reason = REASON_DIVERGED_LINEAR_SOLVE
        @error solver.converged_reason iterations = solver.iterations
        return
      else
        rethrow(e)
      end
    end
    if norm(solver.δU) > solver.dtol
      solver.converged_reason = REASON_DIVERGED_STEP
      @error solver.converged_reason iterations = solver.iterations
      return
    end
    solver.U .+= solver.δU
    @views begin
      solver.residual_fn!(solver.fem, solver.U[:], solver.R[:], solver.ctx)
    end
    norm_r = norm(dofs(solver.fem, solver.R))
    solver.iterations += 1
  end
  if solver.iterations >= solver.maxits
    solver.converged_reason = REASON_DIVERGED_MAXITS
    @warn solver.converged_reason iterations = solver.iterations
  end
  @logmsg ll_resid "  final residual norms:" relative = norm_r / norm_r0 absolute = norm_r
  return
end


# """Use Newton's method (`:standard` or `:modified`) with fixed timestepping to solve a nonlinear finite element problem"""
# function newton(fem::FEMModel, residual, dresidual; type=:standard, t0=0.0, max_time=1.0, dt=0.1, kwargs...)
#   if !(type in (:standard, :modified))
#     error("type must be :standard or :modified")
#   end


#   # Initialize result
#   num_steps = Int((max_time - t0) / dt) + 1
#   fem.time = t0
#   res = NewtonResult(fem; numsteps=num_steps, maxits=maxits)
#   for n in 1:num_steps
#     @debug format("time step n = {} (t = {:0.2g})", n, gettime(fem))
#     if n > 1
#       res.dₙᵏ[n, 1, :] = res.d[n-1, :]
#     end
#     applydirichletboundaries!(fem, @view res.dₙᵏ[n, 1, :])


#     res.num_its[n] = k
#     res.d[n, :] = res.dₙᵏ[n, k, :]

#     # Apply any post processing functions
#     postprocess!(fem, res.d[n, :], res)
#     if n < num_steps
#       step!(fem, dt)
#     end
#   end
#   trim!(res)
#   @info format("Converged with {} Newton-Raphson in {} steps and an average of {:0.1f} Newton iterations per step", type, numsteps(res), sum(res.num_its) / numsteps(res))
#   return res
# end
