export LinearElastoPlasticity1D, has_state, initialize_state!

"""1D linear elastoplastic material
"""
Base.@kwdef struct LinearElastoPlasticity1D <: AbstractMaterial
  """Young's modulus"""
  E::Float64
  """Isotropic hardening modulus"""
  Hκ::Float64
  """Kinematic hardening modulus"""
  Hα::Float64
  """Initial isotropic hardening ISV"""
  κ₀::Float64
  """Initial kinematic hardening ISV"""
  α₀::Float64
  """Consistant elastoplastic tangent modulus"""
  Eep::Float64 = E * (Hκ + Hα) / (E + Hκ + Hα)
  """Constant density, kg/m³"""
  ρ::Float64 = 1.0
end

"""Compute the element internal force for linear elastoplasticity"""
function _internalforce_plasticity_el(fem::FEMModel, u::ElementVector, material::AbstractMaterial, ctx)
  A = get(fem.data, :A, 3e-4)
  σ = getqdata_el!(fem, :σ)

  dxdξ = length(element(u)) / 2
  qfunc = (ξ, i) -> begin
    B = ∇N(fem.P, ξ)' * (1 / dxdξ) # B = dN/dx = ∇N * J^-1
    σ_i = σ[elementindex(mesh(fem), element(u))][i]
    return B' * σ_i * A * dxdξ
  end
  integrate(fem.Q, qfunc, :index)
end

"""Compute the element tangent stiffness matrix for linear elastoplasticity"""
function _dinternalforce_plasticity_el(fem::FEMModel, u::ElementVector, material::AbstractMaterial, ctx)
  A = get(fem.data, :A, 3e-4)
  ∂σ∂ϵ = getqdata_el!(fem, :∂σ∂ϵ)

  dxdξ = length(element(u)) / 2 # dx/dξ
  qfunc = (ξ, i) -> begin
    B = ∇N(fem.P, ξ)' * (1 / dxdξ) # B = dN/dx = ∇N * J^-1
    dϵdu = B
    return B' * ∂σ∂ϵ[elementindex(mesh(fem), element(u))][i] * dϵdu * A * dxdξ
  end
  integrate(fem.Q, qfunc, :index)
end

σ(material::LinearElastoPlasticity1D, _) = error("Use getfemfield_el! to get σ")
∂σ∂ϵ(material::LinearElastoPlasticity1D, _) = error("Use getfemfield_el! to get ∂σ∂ϵ")
get_internalforce_el_fn(::LinearElastoPlasticity1D) = _internalforce_plasticity_el
get_dinternalforce_el_fn(::LinearElastoPlasticity1D) = _dinternalforce_plasticity_el
has_state(::LinearElastoPlasticity1D) = true

"""
Update state variables at quadrature points within an element for the current time step using 
  the current/prior solution at nodes and prior state variables at quadrature points.
"""
function update_state_el!(material::LinearElastoPlasticity1D, fem::FEMModel, u::ElementVector, ctx)
  u_n = getfemfield_el!(fem, :u_n)
  el = elementindex(mesh(fem), element(u))
  σ_n, σ, ∂σ∂ϵ = getqdata_el!(fem, :σ_n), getqdata_el!(fem, :σ), getqdata_el!(fem, :∂σ∂ϵ)
  κ_n, κ = getqdata_el!(fem, :κ_n), getqdata_el!(fem, :κ)
  α_n, α = getqdata_el!(fem, :α_n), getqdata_el!(fem, :α)
  γ_n, γ = getqdata_el!(fem, :γ_n), getqdata_el!(fem, :γ)

  Δu = u.vector - u_n[el].vector
  dxdξ = length(element(u)) / 2 # dx/dξ
  for (i, ξ) in enumerate(nodes(fem.Q))
    B = ∇N(fem.P, ξ)' * (1 / dxdξ) # B = dN/dx = ∇N * J^-1
    Δϵ = B * Δu # dΔu/dx, strain
    σ_tr = σ_n[el][i] + material.E * Δϵ
    relative_stress = σ_tr - α_n[el][i]
    signsigalph = sign(relative_stress)

    f_tr = abs(relative_stress) - κ_n[el][i]
    if f_tr > 1e-5 # plastic 
      Δγ = f_tr / (material.E + material.Hκ + material.Hα)
      σ[el][i] = σ_tr - material.E * Δγ * signsigalph
      α[el][i] = α_n[el][i] + material.Hα * Δγ * signsigalph
      κ[el][i] = κ_n[el][i] + material.Hκ * Δγ
      γ[el][i] = γ_n[el][i] + Δγ
      ∂σ∂ϵ[el][i] = material.Eep
    else # elastic
      σ[el][i] = σ_tr
      α[el][i] = α_n[el][i]
      κ[el][i] = κ_n[el][i]
      γ[el][i] = γ_n[el][i]
      ∂σ∂ϵ[el][i] = material.E
    end
  end
end

"""Save state variables from the current time step"""
function save_state!(::LinearElastoPlasticity1D, fem::FEMModel, U, ctx)
  for el in eachindex(elements(fem.mesh))
    restrict!(getfemfield_el!(fem, :u_n)[el], U)
    for name in (:σ, :α, :κ, :γ)
      getqdata_el!(fem, Symbol(name, "_n"))[el] .= getqdata_el!(fem, name)[el]
    end
  end
end

"""Initialize state variables"""
function initialize_state!(material::LinearElastoPlasticity1D, fem::FEMModel)
  getqdata_el!(fem, :α)
  getqdata_el!(fem, :κ)
  getqdata_el!(fem, :γ)
  getqdata_el!(fem, :γ_n)
  α_n = getqdata_el!(fem, :α_n)
  κ_n = getqdata_el!(fem, :κ_n)
  for el in eachindex(elements(fem.mesh))
    α_n[el][:] .= material.α₀
    κ_n[el][:] .= material.κ₀
  end
end
