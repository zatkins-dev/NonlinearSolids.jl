using LinearAlgebra

function make_f_arc(K̃, d̃; b=0.5)
  den = d̃ ⋅ diagm(K̃) ⋅ d̃
  return (Δd, Δλ) -> √((1 - b) * (Δd ⋅ diagm(K̃) ⋅ Δd) / den + b * Δλ^2)
end

function make_df_arc(K̃, d̃; b=0.5)
  f = make_f_arc(K̃, d̃; b=b)
  den = d̃ ⋅ diagm(K̃) ⋅ d̃
  function df_arc(Δd, Δλ)
    return [((1 - b) * Δd ⋅ diagm(K̃) / (f(Δd, Δλ) * den)) (b / f(Δd, Δλ) * Δλ)]
  end
  return df_arc
end

function newtonarclength(Fint, ∂Fint, dim=1; stop=nothing, b=0.5, Δa=0.1, Fext=8, kwargs...)
  # Read arguments
  options = Dict(kwargs...)
  rtol = pop!(options, :rtol, 1e-16)
  ftol = pop!(options, :ftol, 1e-16)
  stol = pop!(options, :ftol, 0)
  maxsteps = pop!(options, :maxsteps, 20)
  maxits = pop!(options, :maxits, 100)
  for (key, _) in options
    printstyled("warning: unknown option '$key'")
  end

  # Initialize result
  res = ArcLengthResult(dim; maxsteps, maxits)
  Fext = ensurevec(Fext) # Make Fext a vector
  K̃₀ = ∂Fint(res.d[1, :]) # Initial tangent
  d̃₀ = ensurevec(K̃₀ \ Fext)  # Initial displacement increment
  f_arc = make_f_arc(K̃₀, d̃₀; b=b) # Arc length function
  df_arc = make_df_arc(K̃₀, d̃₀; b=b) # Arc length derivative

  # Iterate over steps
  for n in 1:maxsteps+1
    println("\nn = $n")
    # initialization for arc length procedure within time step
    if n < 3 # procedure 1 for first three steps
      d̃ₙ = ensurevec(∂Fint(res.d[n, :]) \ Fext)
      f_tilda_n = f_arc(d̃ₙ, 1)
      Δλ = Δa / f_tilda_n
      Δd = Δλ * d̃ₙ
      res.dₙᵏ[n+1, 1, :] = res.d[n, :] + Δd
      res.λₙᵏ[n+1, 1] = res.λ[n] + Δλ
    else # procedure 2, Lagrange polynomials
      coefs = [1 -3 3]
      res.λₙᵏ[n+1, 1] = coefs ⋅ res.λ[n-2:n]
      res.dₙᵏ[n+1, 1, :] = ensurevec(coefs ⋅ res.d[n-2:n, :])
      Δλ = res.λₙᵏ[n+1, 1] - res.λ[n]
      Δd = res.dₙᵏ[n+1, 1, :] - res.d[n, :]
    end

    r = [
      res.λₙᵏ[n+1, 1] * Fext - Fint(res.dₙᵏ[n+1, 1, :])
      Δa - f_arc(Δd, Δλ)
    ]
    norm_r0 = norm(r[1:end-1])
    if norm_r0 < eps(Float64)
      norm_r0 = 1
    end
    println("norm(r⁰) = $norm_r0")
    res.res_d[n+1, 1] = norm_r0
    res.res_λ[n+1, 1] = r[end]
    k = 1 # Initial iteration count
    # Iterate until residual is small enough
    reason = "diverged: maxits"
    while k < maxits
      # Compute residual
      if res.res_d[n+1, k] < rtol * norm_r0 && res.res_λ[n+1, k] / Δa < ftol
        reason = "converged: rtol and ftol"
        break
      end
      # Compute tangent
      Kᵏ = [∂Fint(res.dₙᵏ[n+1, k, :]) -Fext; df_arc(Δd, Δλ)]
      # Compute displacement increment
      δ = Kᵏ \ vec(r)
      if norm(δ[1:end-1]) < stol && abs(δ[end]) < stol
        reason = "converged: stol"
        break
      end
      # Update displacement and load steps
      res.dₙᵏ[n+1, k+1, :], Δd = [res.dₙᵏ[n+1, k, :], Δd] .+ Ref(δ[1:end-1])
      res.λₙᵏ[n+1, k+1], Δλ = [res.λₙᵏ[n+1, k], Δλ] .+ δ[end]
      # Compute residual
      r = [
        res.λₙᵏ[n+1, k+1] * Fext - Fint(res.dₙᵏ[n+1, k+1, :])
        Δa - f_arc(Δd, Δλ)
      ]
      res.res_d[n+1, k+1] = norm(r[1:end-1])
      res.res_λ[n+1, k+1] = r[end]
      # Update iteration count
      k += 1
    end
    println("time step $n: $reason in $k iterations")
    # Store number of iterations and converged solution
    res.num_its[n+1] = k
    res.d[n+1, :] = res.dₙᵏ[n+1, k, :]
    res.λ[n+1] = res.λₙᵏ[n+1, k]
    res.num_steps = n + 1
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
