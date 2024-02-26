export LinearElasticity1D

"""1D linear elasticity material

This material model is defined by the following stress-strain relationship:

    σ(ϵ) = E * ϵ

where `E` is the Young's modulus.
"""
Base.@kwdef struct LinearElasticity1D <: AbstractMaterial
  """Young's modulus"""
  E::Float64
  """Constant density, kg/m³"""
  ρ::Float64 = 1.0
end

σ(material::LinearElasticity1D, ϵ) = material.E * ϵ
∂σ∂ϵ(material::LinearElasticity1D, _) = material.E
get_internalforce_el_fn(::LinearElasticity1D) = _internalforce_el
get_dinternalforce_el_fn(::LinearElasticity1D) = _dinternalforce_el
