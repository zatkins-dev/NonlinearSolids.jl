export ensurevec, Ramp

"""
Ensure that the input is a vector.
If it is a scalar value, it is converted to a vector
"""
function ensurevec(x)
  return x isa AbstractArray ? vec(x) : vec([x])
end

struct Ramp <: Function
  v0::Float64
  vf::Float64
  t0::Float64
  tf::Float64
  Ramp(vf, tf) = new(0, vf, 0, tf)
end

function (r::Ramp)(t::Float64)
  t < r.tf ? r.v0 + (r.vf - r.v0) * (t - r.t0) / (r.tf - r.t0) : r.vf
end

const Optional{T} = Union{T,Missing} where {T}
