export Jacobian

struct Jacobian <: Function
  material::AbstractMaterial
  dexternalforces::Vector{Function}
  Jacobian(material, dexternalforces=Function[]) = new(material, dexternalforces)
end

"""Static/Quasistatic Jacobian function, U part"""
function (jacobian::Jacobian)(fem::FEMModel, U, K, ctx)
  applydirichletboundaries!(fem, U)

  K .= 0
  if !is_velocitydependent(typeof(jacobian.material))
    compute_dinternalforce(jacobian.material, fem, U, K, ctx; mode=:add)
  end
  for f in jacobian.dexternalforces
    if !is_velocitydependent(typeof(f))
      f(fem, U, K, ctx; mode=:add)
    end
  end
end

"""Static/Quasistatic Jacobian function, U̇ part"""
function (jacobian::Jacobian)(fem::FEMModel, U, U̇, K, ctx)
  # Call velocity-independent Jacobian function
  jacobian(fem, U, K, ctx)

  # Call velocity-independent Jacobian function
  if is_velocitydependent(typeof(jacobian.material))
    compute_dinternalforce(jacobian.material, fem, U, U̇, K, ctx; mode=:add)
  end
  for f in jacobian.dexternalforces
    if is_velocitydependent(typeof(f))
      f(fem, U, U̇, K, ctx; mode=:add)
    end
  end
end


"""Dynamic Jacobian function"""
function (jacobian::Jacobian)(fem::FEMModel, U, U̇, U̇̇, K, ctx::AbstractTimeStepper)
  # Call static Jacobian function
  jacobian(fem, U, U̇, K, ctx)

  # Scale
  K .*= ∂U̇̇∂U(ctx)

  # Add inertia term
  assemblemass!(jacobian.material, fem, K)
end
