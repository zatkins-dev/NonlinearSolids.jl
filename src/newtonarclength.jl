using LinearAlgebra

function make_f_arc(K̃, d̃; b=0.5)
  return (Δd, Δλ) -> first(sqrt((1 - b) * (Δd' * diagm(K̃) * Δd) / (d̃' * diagm(K̃) * d̃) .+ b * Δλ^2))
end

function make_df_arc(K̃, d̃; b=0.5)
  function df_arc(Δd, Δλ)
    f = make_f_arc(K̃, d̃; b=b)
    return [(1 - b) / (f(Δd, Δλ) * (d̃' * diagm(K̃) * d̃)) * Δd' * diagm(K̃) b / f(Δd, Δλ) * Δλ]
  end
  return df_arc
end

function newtonarclength(Fint, ∂Fint, dim=1; stop=nothing, b=0.5, Δa=0.1, Fext=8, rtol=1e-16, ftol=1e-16, maxsteps=20, maxits=100)::ArcLengthResult
  res = ArcLengthResult(dim; maxsteps, maxits)
  K̃₀ = ∂Fint(res.d[1, :]) # Initial tangent
  d̃₀ = Fext \ K̃₀ # Initial displacement increment
  f_arc = make_f_arc(K̃₀, d̃₀; b=b) # Arc length function
  df_arc = make_df_arc(K̃₀, d̃₀; b=b) # Arc length derivative
  Fext = Fext isa Number ? [Fext] : Fext # Make Fext a vector

  # Iterate over steps
  for n in 1:maxsteps
    println("\nn = $n")
    d̃ₙ = Fext \ ∂Fint(res.d[n, :])
    if d̃ₙ isa Number
      d̃ₙ = [d̃ₙ]
    end
    Δλ = 0
    Δd = vec(zeros(dim))

    # initialization for arc length procedure within time step
    if n <= 3 # procedure 1 for first three steps
      f_tilda_n = f_arc(d̃ₙ, 1)
      Δλ = Δa / f_tilda_n
      Δd = Δλ * d̃ₙ
      res.dₙᵏ[n, 1, :] = n == 1 ? zeros(size(Δd)) : res.d[n-1, :] + Δd
      res.λₙᵏ[n, 1] = n == 1 ? 0 : res.λ[n-1] + Δλ
    else # procedure 2, Lagrange polynomials
      res.λₙᵏ[n, 1] = res.λ[n-3] - 3res.λ[n-2] + 3res.λ[n-1]
      res.dₙᵏ[n, 1, :] = res.d[n-3, :] - 3res.d[n-2, :] + 3res.d[n-1, :]
      Δλ = res.λₙᵏ[n, 1] - res.λ[n-1]
      Δd = res.dₙᵏ[n, 1, :] - res.d[n-1, :]
    end

    norm_r0 = norm(Fint(res.dₙᵏ[n, 1, :]) - res.λₙᵏ[n, 1] * Fext)
    if norm_r0 < eps(Float64)
      norm_r0 = 1
    end
    k = 1 # Initial iteration count
    # Iterate until residual is small enough
    while k < maxits
      # println("k = $(k)")
      r = [res.λₙᵏ[n, k] * Fext - Fint(res.dₙᵏ[n, k, :]) Δa - f_arc(Δd, Δλ)]
      res.res_d[n, k] = norm(r[1:end-1]) / norm_r0
      res.res_λ[n, k] = r[end] / Δa
      if res.res_d[n, k] < rtol && res.res_λ[n, k] < ftol
        break
      end
      # println("r = $r, d = $(res.dₙᵏ[n, k, :]), λ = $(res.λₙᵏ[n, k])")
      # Compute tangent
      Kᵏ = [∂Fint(res.dₙᵏ[n, k, :]) -Fext; df_arc(Δd, Δλ)]
      # println("K = $(Kᵏ)")
      # Compute displacement increment
      δ = Kᵏ \ vec(r)
      res.dₙᵏ[n, k+1, :] = res.dₙᵏ[n, k, :] + δ[1:end-1]
      res.λₙᵏ[n, k+1] = res.λₙᵏ[n, k] + δ[end]
      # Update step size
      Δd = Δd + δ[1:end-1]
      Δλ = Δλ + δ[end]
      # Update iteration count
      k += 1
    end
    println("time step $n converged in $k iterations")
    # Store number of iterations
    res.num_its[n] = k
    # Store solution at time step
    res.d[n, :] = res.dₙᵏ[n, k, :]
    res.λ[n] = res.λₙᵏ[n, k]
    res.num_steps = n
    println("d = $(res.d[n, :]), λ = $(res.λ[n])")
    if !isnothing(stop) && stop(res.d[n, :], res.λ[n])
      println("dmax exceeded at step $n, stopping")
      break
    end
  end
  # Trim arrays
  trim!(res)
  # Return solution
  return res
end
