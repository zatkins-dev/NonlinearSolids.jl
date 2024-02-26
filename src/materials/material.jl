export AbstractMaterial, σ, ∂σ∂ϵ, density, get_internalforce_el_fn, get_dinternalforce_el_fn, is_velocitydependent

abstract type AbstractMaterial end

density(material::AbstractMaterial) = material.ρ
σ(material::AbstractMaterial, ϵ) = error("Not implemented")
∂σ∂ϵ(material::AbstractMaterial, ϵ) = error("Not implemented")
get_internalforce_el_fn(material::AbstractMaterial) = error("Not implemented")
get_dinternalforce_el_fn(material::AbstractMaterial) = error("Not implemented")
is_velocitydependent(material::AbstractMaterial) = false
