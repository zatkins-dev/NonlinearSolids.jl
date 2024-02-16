using LinearAlgebra

export newtonraphson, modifiednewton

function newton(Fint, ∂Fint, dim=1; type=:standard, Fext=8, ΔFext=Fext / 32, kwargs...)
  if !(type in (:standard, :modified))
    error("type must be :standard or :modified")
  end
  # Read arguments
  options = Dict(kwargs...)
  rtol = pop!(options, :rtol, 1e-6)
  stol = pop!(options, :stol, 1e-12)
  atol = pop!(options, :atol, 1e-12)
  dtol = pop!(options, :dtol, 1e6)
  maxits = pop!(options, :maxits, 100)
  for (key, _) in options
    @warn "warning: unknown option '$key'"
  end
  @info """
Running $(type) Newton-Raphson with options:
  atol: $atol
  rtol: $rtol
  dtol: $dtol
  maxits: $maxits
"""

  # Initialize result
  num_steps = Int(Fext ÷ ΔFext)
  res = NewtonResult(dim; numsteps=num_steps, maxits=maxits)
  Fextₙ = [n * ensurevec(ΔFext) for n in 1:num_steps] # Array for forces at each time step

  # Iterate over steps
  for n in 1:num_steps
    @debug format("time step n = {} (t = {:0.2g})", n, gettime(fem))
    if n > 1
      res.dₙᵏ[n, 1, :] = res.d[n-1, :]
    end
    r = ensurevec(Fextₙ[n] - Fint(res.dₙᵏ[n, 1, :])) # Initial residual
    Kᵏ = ∂Fint(res.dₙᵏ[n, 1, :]) # Initial tangent
    norm_r0 = norm(r) > eps() ? norm(r) : 1
    res.res_d[n, 1] = norm_r0
    δd⁰ = 0
    k = 1 # Initial iteration count
    # Iterate until residual is small enough
    while k < maxits
      if res.res_d[n, k] / norm_r0 < rtol
        @debug "  converged: rtol in $k iterations"
        break
      elseif res.res_d[n, k] < atol
        @debug "  converged: atol in $k iterations"
        break
      end
      # Compute tangent
      if type == :standard
        Kᵏ = ∂Fint(res.dₙᵏ[n, k, :])
      end
      # Compute displacement increment
      δdᵏ = ensurevec(Kᵏ \ r)
      if k == 1
        δd⁰ = δdᵏ
      end
      if norm(δdᵏ) > dtol
        @error "  diverged Newton step at step n=$n, k=$k"
        trim!(res, n - 1)
        return res
      elseif norm(δdᵏ) / norm(δd⁰) < stol
        @debug "  converged: stol in $k iterations"
        break
      end
      # Update displacement
      res.dₙᵏ[n, k+1, :] = res.dₙᵏ[n, k, :] + δdᵏ
      # Compute residual
      r = Fextₙ[n] - Fint(res.dₙᵏ[n, k+1, :])
      res.res_d[n, k+1] = norm(r)
      # Update iteration count
      k += 1
    end
    if k >= maxits
      @warn "  diverged max its ($k iterations)"
    end
    @debug "  final errors:" relative = res.res_d[n, k] / norm_r0 absolute = res.res_d[n, k]
    # Store number of iterations and converged solution
    res.num_its[n] = k
    res.d[n, :] = res.dₙᵏ[n, k, :]
  end
  # Return solution
  trim!(res)
  return res
end

include("newton_fem.jl")

function newtonraphson(args...; kwargs...)
  return newton(args...; type=:standard, kwargs...)
end

function modifiednewton(args...; kwargs...)
  return newton(args...; type=:modified, kwargs...)
end
