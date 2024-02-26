"""Default implementation of the element internal force function"""
function _internalforce_el(fem::FEMModel, u::ElementVector, material::AbstractMaterial, ctx)
  A = get(fem.data, :A, 3e-4)

  dxdξ = length(element(u)) / 2
  qfunc = (ξ) -> begin
    B = ∇N(fem.P, ξ)' * (1 / dxdξ) # B = dN/dx = ∇N * J^-1
    ϵ = B ⋅ u.vector # du/dx, strain
    return B' * σ(material, ϵ) * A * dxdξ
  end
  integrate(fem.Q, qfunc)
end

"""Default implementation of the element tangent stiffness matrix function"""
function _dinternalforce_el(fem::FEMModel, u::ElementVector, material::AbstractMaterial, ctx)
  A = get(fem.data, :A, 3e-4)

  dxdξ = length(element(u)) / 2 # dx/dξ
  qfunc = (ξ) -> begin
    B = ∇N(fem.P, ξ)' * (1 / dxdξ) # B = dN/dx = ∇N * J^-1
    ϵ = B ⋅ u.vector # du/dx, strain
    dϵdu = B
    return B' * ∂σ∂ϵ(material, ϵ) * dϵdu * A * dxdξ
  end
  integrate(fem.Q, qfunc)
end

"""Default implementation of the global internal force function"""
function compute_internalforce(material::AbstractMaterial, fem::FEMModel, U, Fint, ctx; mode=:add)
  if !(mode in (:add, :set))
    throw(ArgumentError("Invalid mode: $mode"))
  end
  if mode == :set
    Fint .= 0
  end
  d_el = getfemfield_el!(fem, :d)
  f_int_el = getfemfield_el!(fem, :f_int)
  for el in eachindex(elements(fem.mesh))
    restrict!(d_el[el], U)
    f_int_el[el].vector .= get_internalforce_el_fn(material)(fem, d_el[el], material, ctx)
    unrestrict!(f_int_el[el], Fint)
  end
end
"""Default implementation of the global internal force function, rate-dependent"""
function compute_internalforce(material::AbstractMaterial, fem::FEMModel, U, U̇, Fint, ctx; mode=:add)
  if !(mode in (:add, :set))
    throw(ArgumentError("Invalid mode: $mode"))
  end
  if mode == :set
    Fint .= 0
  end
  d_el = getfemfield_el!(fem, :d)
  v_el = getfemfield_el!(fem, :v)
  f_int_el = getfemfield_el!(fem, :f_int)
  for el in eachindex(elements(fem.mesh))
    restrict!(d_el[el], U)
    restrict!(v_el[el], U̇)
    f_int_el[el].vector .= get_internalforce_el_fn(material)(fem, d_el[el], v_el[el], material, ctx)
    unrestrict!(f_int_el[el], Fint)
  end
end

"""Default implementation of the global tangent stiffness matrix function"""
function compute_dinternalforce(material::AbstractMaterial, fem::FEMModel, U, K, ctx; mode=:add)
  if !(mode in (:add, :set))
    throw(ArgumentError("Invalid mode: $mode"))
  end
  if mode == :set
    K .= 0
  end
  d_el = getfemfield_el!(fem, :d)
  K_el = getfemfield_el!(fem, :K, ismat=true)
  for el in eachindex(elements(fem.mesh))
    restrict!(d_el[el], U)
    K_el[el].matrix .= get_dinternalforce_el_fn(material)(fem, d_el[el], material, ctx)
    assemble!(K_el[el], K)
  end
end

"""Default implementation of the global tangent stiffness matrix function, rate-dependent"""
function compute_dinternalforce(material::AbstractMaterial, fem::FEMModel, U, U̇, K, ctx; mode=:add)
  if !(mode in (:add, :set))
    throw(ArgumentError("Invalid mode: $mode"))
  end
  if mode == :set
    K .= 0
  end
  d_el = getfemfield_el!(fem, :d)
  v_el = getfemfield_el!(fem, :v)
  K_el = getfemfield_el!(fem, :K, ismat=true)
  for el in eachindex(elements(fem.mesh))
    restrict!(d_el[el], U)
    restrict!(v_el[el], U̇)
    K_el[el].matrix .= get_dinternalforce_el_fn(material)(fem, d_el[el], v_el[el], material, ctx)
    assemble!(K_el[el], K)
  end
end
