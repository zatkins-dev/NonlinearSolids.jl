using LinearAlgebra

export newtonarclength

"""Arc length functor"""
struct ArcLength <: Function
  denominator::Real
  K̃diag::Diagonal
  b::Real
  function ArcLength(K̃, d̃; b=0.5)
    K̃₀ = reshape(K̃, length(d̃), length(d̃))
    K̃diag = Diagonal(diag(K̃₀))
    den = d̃' * K̃diag * d̃
    new(den, K̃diag, b)
  end
  function (f::ArcLength)(Δd, Δλ)
    return √((1 - f.b) * (Δd' * f.K̃diag * Δd) / f.denominator + f.b * Δλ^2)
  end
end

"""Arc length derivative functor"""
∇(f::ArcLength) = (Δd, Δλ) -> [((1 - f.b) * Δd' * f.K̃diag / (f(Δd, Δλ) * f.denominator)) (f.b / f(Δd, Δλ) * Δλ)]

function make_f_arc(K̃, d̃; b=0.5)
  K̃₀ = reshape(K̃, length(d̃), length(d̃))
  K̃diag = Diagonal(diag(K̃₀))
  den = d̃' * K̃diag * d̃
  return (Δd, Δλ) -> √((1 - b) * (Δd' * K̃diag * Δd) / den + b * Δλ^2)
end

function make_df_arc(K̃, d̃; b=0.5)
  f = make_f_arc(K̃, d̃; b=b)
  K̃₀ = reshape(K̃, length(d̃), length(d̃))
  K̃diag = Diagonal(diag(K̃₀))
  den = d̃' * K̃diag * d̃
  function df_arc(Δd, Δλ)
    return [((1 - b) * Δd' * K̃diag / (f(Δd, Δλ) * den)) (b / f(Δd, Δλ) * Δλ)]
  end
  return df_arc
end

function newtonarclength(Fint::Function, ∂Fint::Function, dim::Int=1; stop=nothing, b=0.5, Δa=0.1, Fext=8, kwargs...)
  # Read arguments
  options = Dict(kwargs...)
  rtol = pop!(options, :rtol, 1e-16)
  ftol = pop!(options, :ftol, 1e-16)
  stol = pop!(options, :stol, 0)
  maxsteps = pop!(options, :maxsteps, 20)
  maxits = pop!(options, :maxits, 100)
  for (key, _) in options
    @warn "unknown option '$key'"
  end
  @info """
  Running Newton-Raphson Arc Length with options:
    Δa: $Δa
    b: $b
    rtol: $rtol
    ftol: $ftol
    stol: $stol
    maxsteps: $maxsteps
    maxits: $maxits
  """

  # Initialize result
  res = ArcLengthResult(dim; maxsteps=maxsteps, maxits=maxits)
  Fext = ensurevec(Fext) # Make Fext a vector
  K̃₀ = ∂Fint(res.d[1, :]) # Initial tangent
  d̃₀ = ensurevec(K̃₀ \ Fext)  # Initial displacement increment
  f_arc = ArcLength(K̃₀, d̃₀; b=b) # Arc length function

  # Iterate over steps
  for n in 1:maxsteps+1
    @debug format("time step n = {} (t = {:0.2g})", n + 1, gettime(fem))
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
    res.res_d[n+1, 1] = norm_r0
    res.res_λ[n+1, 1] = r[end]
    k = 1 # Initial iteration count
    # Iterate until residual is small enough
    while k < maxits
      # Compute residual
      if res.res_d[n+1, k] < rtol * norm_r0 && res.res_λ[n+1, k] / Δa < ftol
        @debug "  converged: rtol and ftol in $k iterations"
        break
      end
      # Compute tangent
      Kᵏ = [∂Fint(res.dₙᵏ[n+1, k, :]) -Fext; ∇(f_arc)(Δd, Δλ)]
      # Compute displacement increment
      δ = Kᵏ \ vec(r)
      if norm(δ[1:end-1]) < stol && abs(δ[end]) < stol
        @debug "  converged: stol in $k iterations"
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
    if k >= maxits
      @warn "  diverged max its ($k iterations)"
    end
    @info format("""
  final relative errors:
    ||rₙ - r⁰ₙ|| / ||r⁰ₙ|| = {:0.3g}
    (Δa - f(λ))  /   Δa    = {:0.3g}
""", res.res_d[n+1, k] / norm_r0, res.res_λ[n+1, k] / Δa)
    # Store number of iterations and converged solution
    res.num_its[n+1] = k
    res.d[n+1, :] = res.dₙᵏ[n+1, k, :]
    res.λ[n+1] = res.λₙᵏ[n+1, k]
    res.num_steps = n + 1
    # Check stopping criterion
    if !isnothing(stop) && stop(res.d[n+1, :], res.λ[n+1])
      @debug("Stopping criterion met at step $(n+1), stopping")
      break
    end
  end
  # Trim arrays
  trim!(res)
  # Return solution
  return res
end

include("newtonarclength_fem.jl")
