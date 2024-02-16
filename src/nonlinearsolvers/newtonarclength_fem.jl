function newtonarclength(fem::FEMModel, Fint::Function, ∂Fint∂d::Function, Fext::Function; stop=nothing, b=0.5, Δa=0.1, kwargs...)
  # Read arguments
  options = Dict(kwargs...)
  rtol = pop!(options, :rtol, 1e-16)
  ftol = pop!(options, :ftol, 1e-16)
  stol = pop!(options, :stol, 0)
  maxsteps = pop!(options, :maxsteps, 20)
  maxits = pop!(options, :maxits, 100)
  for (key, _) in options
    printstyled("warning: unknown option '$key'")
  end

  println("Running Newton-Raphson Arc Length with options:")
  println("  Δa: $Δa")
  println("  b: $b")
  println("  rtol: $rtol")
  println("  ftol: $ftol")
  println("  stol: $stol")
  println("  maxsteps: $maxsteps")
  println("  maxits: $maxits")
  println(fem)

  # Initialize result and vectors
  res = ArcLengthFEMResult(fem; maxsteps=maxsteps, maxits=maxits)
  K̃₀ = zeros(numdof(fem), numdof(fem)) # Reference consistent displacement tangent
  Kd = zeros(numdof(fem), numdof(fem)) # Current consistent displacement tangent
  d = reshape(res.d[1, :], :) # Flattened view of displacement vector
  Δd = zeros(length(dofs(fem, d))) # Displacement step
  Δλ = 0 # Load step
  δ = zeros(length(dofs(fem, d)) + 1) # Incremental displacement and load factor
  r = zeros(length(dofs(fem, d)) + 1) # Residual (excluding constrained DoFs)
  F_ext = zeros(length(d)) # External force vector
  F_int = zeros(length(d)) # Internal force vector

  # Compute initial values
  ∂Fint∂d(fem, d, K̃₀) # Initial tangent
  Fext(fem, F_ext) # External force
  d̃₀ = dofs(fem, K̃₀) \ dofs(fem, F_ext) # Initial displacement increment
  f_arc = make_f_arc(dofs(fem, K̃₀), d̃₀; b=b) # Arc length function
  df_arc = make_df_arc(dofs(fem, K̃₀), d̃₀; b=b) # Arc length derivative
  detKlast = det(dofs(fem, K̃₀)) # Previous det(K̃ₙ) for initialization stage 1
  signΔλlast = 1 # Previous sign of Δλ for initialization stage 1

  # Iterate over steps
  for n in 1:maxsteps-1
    println("\nn = $(n+1)")
    # Initialization for arc length procedure within time step
    d = reshape(res.d[n, :], :)
    if n < 3 # procedure 1 for first three steps
      ∂Fint∂d(fem, d, Kd)
      d̃ₙ = dofs(fem, Kd) \ dofs(fem, F_ext)
      f_tilda_n = f_arc(d̃ₙ, 1)
      Δλ = Δa / f_tilda_n
      # determine sign of Δλ
      detKn = det(dofs(fem, Kd))
      signΔλ = signΔλlast
      if sign(detKn) == -sign(detKlast)
        signΔλ = -signΔλlast
      end
      Δλ = signΔλ * abs(Δλ)
      signΔλlast = signΔλ
      detKlast = detKn
      # update steps
      Δd .= Δλ * d̃ₙ
      res.dₙᵏ[n+1, 1, :, :] .= d + expand(fem, Δd)
      res.λₙᵏ[n+1, 1] = res.λ[n] + Δλ
    else # procedure 2, Lagrange polynomials
      coefs = vec([1 -3 3])
      res.λₙᵏ[n+1, 1] = coefs ⋅ res.λ[n-2:n]
      res.dₙᵏ[n+1, 1, :, :] = coefs' * reshape(res.d[n-2:n, :, :], length(coefs), :)
      Δλ = res.λₙᵏ[n+1, 1] - res.λ[n]
      Δd .= dofs(fem, reshape(res.dₙᵏ[n+1, 1, :, :] - res.d[n, :, :], :))
    end

    # Get flattened view
    d = reshape(res.dₙᵏ[n+1, 1, :, :], :)
    applydirichletboundaries!(fem, d)
    # Compute initial monolithic residual
    Fint(fem, d, F_int)
    r .= [
      dofs(fem, res.λₙᵏ[n+1, 1] * F_ext - F_int)
      Δa - f_arc(Δd, Δλ)
    ]
    norm_r0 = norm(r[1:end-1])
    if norm_r0 < eps(Float64)
      norm_r0 = 1
    end
    println("  norm(r⁰) = $norm_r0")
    res.res_d[n+1, 1] = norm_r0
    res.res_λ[n+1, 1] = r[end]
    k = 1 # Initial iteration count
    # Iterate until residual is small enough
    while k < maxits
      # Compute residual
      if res.res_d[n+1, k] < rtol * norm_r0 && res.res_λ[n+1, k] / Δa < ftol
        printstyled("  converged: rtol and ftol in $k iterations\n", color=:green)
        break
      end
      # Compute tangent
      ∂Fint∂d(fem, d, Kd)
      Kᵏ = [dofs(fem, Kd) -dofs(fem, F_ext); df_arc(Δd, Δλ)]
      # Compute displacement increment
      try
        δ = Kᵏ \ r
      catch e
        if e isa SingularException
          printstyled("  error: diverged linear solve\n", color=:red)
          trim!(res)
          return res
        else
          rethrow(e)
        end
      end
      if norm(δ[1:end-1]) < stol && abs(δ[end]) < stol
        printstyled("  converged: stol in $k iterations\n", color=:green)
        break
      end
      # Update displacement and load steps
      res.dₙᵏ[n+1, k+1, :, :] .= d + expand(fem, δ[1:end-1])
      Δd += δ[1:end-1]
      res.λₙᵏ[n+1, k+1], Δλ = [res.λₙᵏ[n+1, k], Δλ] .+ δ[end]
      # Apply Dirichlet BCs
      d = reshape(res.dₙᵏ[n+1, k+1, :, :], :)
      applydirichletboundaries!(fem, d)
      # Compute monolithic residual
      Fint(fem, d, F_int)
      r = [
        dofs(fem, res.λₙᵏ[n+1, k+1] * F_ext - F_int)
        Δa - f_arc(Δd, Δλ)
      ]
      res.res_d[n+1, k+1] = norm(r[1:end-1])
      res.res_λ[n+1, k+1] = r[end]
      # Update iteration count
      k += 1
    end
    if k >= maxits
      printstyled("  warning: diverged max its ($k iterations)\n", color=:yellow)
    end
    println("  final relative errors:")
    printfmtln("    ||rₙ - r⁰ₙ|| / ||r⁰ₙ|| = {:0.3g}", res.res_d[n, k] / norm_r0)
    printfmtln("    (Δa - f(λ))  /   Δa    = {:0.3g}", res.res_λ[n+1, k] / Δa)

    # Store number of iterations and converged solution
    res.num_its[n+1] = k
    res.d[n+1, :] = res.dₙᵏ[n+1, k, :]
    res.λ[n+1] = res.λₙᵏ[n+1, k]
    res.num_steps = n + 1

    # Apply any postprocessing functions
    step!(fem, Δλ)
    postprocess!(fem, reshape(res.d[n, :, :], :), res)

    # Check stopping criterion
    if !isnothing(stop) && stop(res.d[n+1, :], res.λ[n+1])
      println("Stopping criterion met at step $n, stopping")
      break
    end
  end
  # Trim arrays
  trim!(res)
  # Return solution
  return res
end
