export ExponentialMaterial

"""Exponential nonlinear material model

This material model is defined by the following stress-strain relationship:

    σ(ϵ) = σ_sat * (1 - exp(-b * ϵ))

where `σ_sat` is the saturated stress and `b` is the exponential coefficient.
"""
Base.@kwdef mutable struct ExponentialMaterial <: AbstractMaterial
  """Saturated stress, Pa"""
  σ_sat::Float64 # Pa
  """Exponential coefficient"""
  b::Float64
  """Constant density, kg/m³"""
  ρ::Float64 = 1.0
end

σ(mat::ExponentialMaterial, ϵ) = mat.σ_sat * (1 - exp(-mat.b * ϵ))
∂σ∂ϵ(mat::ExponentialMaterial, ϵ) = mat.b * mat.σ_sat * exp(-mat.b * ϵ)
get_internalforce_el_fn(::ExponentialMaterial) = _internalforce_el
get_dinternalforce_el_fn(::ExponentialMaterial) = _dinternalforce_el
