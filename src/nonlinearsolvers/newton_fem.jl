
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
    printstyled("warning: unknown option '$key'")
  end
  println("Running $(type) Newton-Raphson with options:")
  println("  atol: $atol")
  println("  rtol: $rtol")
  println("  dtol: $dtol")
  println("  maxits: $maxits")
  println(fem)

  # Initialize result
  num_steps = Int((max_time - t0) / dt) + 1
  fem.time = t0
  res = NewtonFEMResult(numdof(fem), dim(fem); num_steps=num_steps, maxits=maxits)
  R = zeros(numdof(fem))
  K = zeros(numdof(fem), numdof(fem))

  for n in 1:num_steps
    updateboundaries!(fem)
    printfmtln("time step n = {} (t = {:0.2g})", n, gettime(fem))
    if n > 1
      res.dₙᵏ[n, 1, :, :] = res.d[n-1, :, :]
    end
    d = reshape(res.dₙᵏ[n, 1, :, :], :)
    applydirichletboundaries!(fem, d)
    residual(fem, d, R)
    dresidual(fem, d, K)
    r = dofs(fem, R)
    res.res_d[n, 1] = norm(r)
    norm_r0 = norm(r) > eps() ? norm(r) : 1
    δd = zeros(size(dofs(fem, d)))
    k = 1 # Initial iteration count
    while k < maxits
      if res.res_d[n, k] / norm_r0 < rtol
        printstyled("  converged: rtol in $k iterations\n", color=:green)
        break
      end
      if res.res_d[n, k] < atol
        printstyled("  converged: atol in $k iterations\n", color=:green)
        break
      end
      if type == :standard
        dresidual(fem, d, K)
      end
      try
        δd = -dofs(fem, K) \ dofs(fem, R)
      catch e
        if e isa SingularException
          printstyled("  error: diverged linear solve\n", color=:red)
          trim!(res, n - 1)
          return res
        else
          rethrow(e)
        end
      end
      if norm(δd) > dtol
        printstyled("  error: diverged Newton step\n", color=:red)
        trim!(res, n - 1)
        return res
      end
      dnew = d + expand(fem, δd)
      applydirichletboundaries!(fem, dnew)
      res.dₙᵏ[n, k+1, :, :] .= dnew
      d = reshape(res.dₙᵏ[n, k+1, :, :], :)
      residual(fem, d, R)
      res.res_d[n, k+1] = norm(dofs(fem, R))
      k += 1
    end
    if k >= maxits
      printstyled("  warning: diverged max its ($k iterations)\n", color=:yellow)
    end
    println("  final errors:")
    printfmtln("    rel: {:0.3g}; abs: {:0.3g}\n", res.res_d[n, k] / norm_r0, res.res_d[n, k])
    res.num_its[n] = k
    res.d[n, :, :] = res.dₙᵏ[n, k, :, :]

    # Apply any post processing functions
    postprocess!(fem, reshape(res.d[n, :, :], :), res)
    if n < num_steps
      step!(fem, dt)
    end
  end
  trim!(res)
  return res
end
