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

function _applymass_el_fn(fem::FEMModel, a::ElementVector, A, ρ)
  J = length(element(a)) / 2 # dx/dξ
  return integrate(fem.Q) do ξ
    N(fem.P, ξ) * N(fem.P, ξ)' * a.vector * A * ρ * J
  end
end

"""Dynamic residual function"""
function (residual::Residual)(fem::FEMModel, U, U̇, U̇̇, R, ctx)
  # Call static residual function
  residual(fem, U, U̇, R, ctx)

  # Add inertia term
  A = get(fem.data, :A, 1.0)
  ρ = density(residual.material)

  a_el = getfemfield_el!(fem, :a)
  m_dot_a_el = getfemfield_el!(fem, :m_dot_a)
  for el in eachindex(elements(fem.mesh))
    restrict!(a_el[el], U̇̇)

    m_dot_a_el[el].vector .= _applymass_el_fn(fem, a_el[el], A, ρ)

    unrestrict!(m_dot_a_el[el], R)
  end
end
