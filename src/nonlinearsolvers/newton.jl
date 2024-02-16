using LinearAlgebra

export newtonraphson, modifiednewton

function newton(Fint, ∂Fint, dim=1; type=:standard, Fext=8, ΔFext=Fext / 32, kwargs...)
  if !(type in (:standard, :modified))
    error("type must be :standard or :modified")
  end
  # Read arguments
  options = Dict(kwargs...)
  rtol = pop!(options, :rtol, 1e-16)
  stol = pop!(options, :stol, 1e-16)
  dtol = pop!(options, :dtol, 1e6)
  maxits = pop!(options, :maxits, 100)
  for (key, _) in options
    printstyled("warning: unknown option '$key'")
  end

  # Initialize result
  num_steps = Int(Fext ÷ ΔFext)
  res = NewtonResult(dim; numsteps=num_steps, maxits=maxits)
  Fextₙ = [n * ensurevec(ΔFext) for n in 1:num_steps] # Array for forces at each time step

  # Iterate over steps
  for n in 1:num_steps
    println("\nn = $n")
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
    reason = "diverged: maxits"
    while k < maxits
      if res.res_d[n, k] < rtol
        reason = "converged: rtol"
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
        reason = "diverged: dtol"
        printstyled("error: diverged solve at time step $n\n", color=:red)
        trim!(res, n - 1)
        return res
      elseif norm(δdᵏ) / norm(δd⁰) < stol
        reason = "converged: stol"
        break
      end
      # Update displacement
      res.dₙᵏ[n, k+1, :] = res.dₙᵏ[n, k, :] + δdᵏ
      # Compute residual
      r = Fextₙ[n] - Fint(res.dₙᵏ[n, k+1, :])
      res.res_d[n, k+1] = norm(r) / norm_r0
      # Update iteration count
      k += 1
    end
    println("time step $n: $reason in $k iterations")
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
