"""Use Newton's method (`:standard` or `:modified`) with fixed timestepping to solve a nonlinear finite element problem"""
function newton(fem::FEMModel, residual, dresidual; type=:standard, t0=0.0, max_time=1.0, dt=0.1, kwargs...)
  if !(type in (:standard, :modified))
    error("type must be :standard or :modified")
  end
  # Read arguments
  options = Dict(kwargs...)
  atol = pop!(options, :atol, 1e-16)
  rtol = pop!(options, :rtol, 1e-16)
  dtol = pop!(options, :dtol, 1e6)
  maxits = pop!(options, :maxits, 100)
  for (key, _) in options
    @warn "unknown option '$key'"
  end
  @info """
Running $(type) Newton-Raphson with options:
  atol: $atol
  rtol: $rtol
  dtol: $dtol
  maxits: $maxits
""" fem

  # Initialize result
  num_steps = Int((max_time - t0) / dt) + 1
  fem.time = t0
  res = NewtonResult(fem; numsteps=num_steps, maxits=maxits)
  R = zeros(numdof(fem))
  K = zeros(numdof(fem), numdof(fem))

  for n in 1:num_steps
    updateboundaries!(fem)
    @debug format("time step n = {} (t = {:0.2g})", n, gettime(fem))
    if n > 1
      res.dₙᵏ[n, 1, :] = res.d[n-1, :]
    end
    applydirichletboundaries!(fem, @view res.dₙᵏ[n, 1, :])
    residual(fem, res.dₙᵏ[n, 1, :], R)
    dresidual(fem, res.dₙᵏ[n, 1, :], K)
    r = dofs(fem, R)
    res.res_d[n, 1] = norm(r)
    norm_r0 = norm(r) > eps() ? norm(r) : 1
    δd = zeros(size(dofs(fem, R)))
    k = 1 # Initial iteration count
    while k < maxits
      if res.res_d[n, k] / norm_r0 < rtol
        @debug "  converged: rtol in $k iterations"
        break
      end
      if res.res_d[n, k] < atol
        @debug "  converged: atol in $k iterations"
        break
      end
      if type == :standard
        dresidual(fem, res.dₙᵏ[n, k, :], K)
      end
      try
        δd = -dofs(fem, K) \ dofs(fem, R)
      catch e
        if e isa SingularException
          @error "  diverged linear solve at step n=$n, k=$k"
          trim!(res, n - 1)
          return res
        else
          rethrow(e)
        end
      end
      if norm(δd) > dtol
        @error "  diverged Newton step at step n=$n, k=$k"
        trim!(res, n - 1)
        return res
      end
      res.dₙᵏ[n, k+1, :] = res.dₙᵏ[n, k, :] + expand(fem, δd)
      applydirichletboundaries!(fem, @view res.dₙᵏ[n, k+1, :])
      residual(fem, res.dₙᵏ[n, k+1, :], R)
      res.res_d[n, k+1] = norm(dofs(fem, R))
      k += 1
    end
    if k >= maxits
      @warn "  diverged max its ($k iterations)"
    end
    @debug "  final errors:" relative = res.res_d[n, k] / norm_r0 absolute = res.res_d[n, k]
    res.num_its[n] = k
    res.d[n, :] = res.dₙᵏ[n, k, :]

    # Apply any post processing functions
    postprocess!(fem, res.d[n, :], res)
    if n < num_steps
      step!(fem, dt)
    end
  end
  trim!(res)
  @info format("Converged with {} Newton-Raphson in {} steps and an average of {:0.1f} Newton iterations per step", type, numsteps(res), sum(res.num_its) / numsteps(res))
  return res
end
