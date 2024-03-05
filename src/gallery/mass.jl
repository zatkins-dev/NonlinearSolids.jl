function _mass_el_fn!(fem::FEMModel, el::Element, A, ρ)
  J = length(el) / 2 # dx/dξ
  return integrate(fem.Q) do ξ
    N(fem.P, ξ) * N(fem.P, ξ)' * A * ρ * J
  end
end

function assemblemass!(material::AbstractMaterial, fem::FEMModel, K)
  A = get(fem.data, :A, 1.0)
  ρ = density(material)

  m = getfemfield_el!(fem, :m; ismat=true)
  for m_el in m
    m_el.matrix .= _mass_el_fn!(fem, element(m_el), A, ρ)
    assemble!(m_el, K)
  end
end

function _applymass_el_fn!(fem::FEMModel, a::ElementVector, A, ρ)
  J = length(element(a)) / 2 # dx/dξ
  return integrate(fem.Q) do ξ
    N(fem.P, ξ) * N(fem.P, ξ)' * a.vector * A * ρ * J
  end
end

function applymass!(material::AbstractMaterial, fem::FEMModel, U̇̇, R)
  A = get(fem.data, :A, 1.0)
  ρ = density(material)

  a_el = getfemfield_el!(fem, :a)
  m_dot_a_el = getfemfield_el!(fem, :m_dot_a)
  for el in eachindex(elements(fem.mesh))
    restrict!(a_el[el], U̇̇)
    m_dot_a_el[el].vector .= _applymass_el_fn!(fem, a_el[el], A, ρ)
    unrestrict!(m_dot_a_el[el], R)
  end
end
