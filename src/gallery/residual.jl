export Residual, is_velocitydependent

is_velocitydependent(::Type) = false

struct Residual <: Function
  material::AbstractMaterial
  externalforces::Vector{Function}
  Residual(material, externalforces=Function[]) = new(material, externalforces)
end

"""Static/Quasistatic residual function, U part"""
function (residual::Residual)(fem::FEMModel, U, R, ctx)
  applydirichletboundaries!(fem, U)
  R .= 0
  Fext = zeros(length(U))
  if !is_velocitydependent(typeof(residual.material))
    compute_internalforce(residual.material, fem, U, R, ctx; mode=:add)
  end
  for f in residual.externalforces
    if !is_velocitydependent(typeof(f))
      f(fem, U, Fext, ctx; mode=:add)
    end
  end
  applyneumannboundaries!(fem, Fext)
  R .-= Fext
end

"""Static/Quasistatic residual function, U̇ part"""
function (residual::Residual)(fem::FEMModel, U, U̇, R, ctx)
  # Call velocity-independent residual function
  residual(fem, U, R, ctx)

  # Call velocity-independent residual function
  if is_velocitydependent(typeof(residual.material))
    compute_internalforce(residual.material, fem, U, U̇, R, ctx; mode=:add)
  end

  Fext = zeros(length(U))
  for f in residual.externalforces
    if is_velocitydependent(typeof(f))
      f(fem, U, U̇, Fext, ctx; mode=:add)
    end
  end
  R .-= Fext
end

"""Dynamic residual function"""
function (residual::Residual)(fem::FEMModel, U, U̇, U̇̇, R, ctx)
  # Call static residual function
  residual(fem, U, U̇, R, ctx)

  # Add inertia term
  applymass!(residual.material, fem, U̇̇, R)
end
