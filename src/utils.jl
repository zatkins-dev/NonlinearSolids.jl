export ensurevec, make_ramp

"""
Ensure that the input is a vector.
If it is a scalar value, it is converted to a vector
"""
function ensurevec(x)
  return x isa AbstractArray ? vec(x) : vec([x])
end

"""Linearly interpolate between two values"""
function make_ramp(v0, vf, t0, tf)
  t_total = tf - t0

  return function ramp(t)
    t < tf ? v0 + (vf - v0) * (t - t0) / t_total : vf
  end
end

function make_ramp(vf, tf)
  return make_ramp(0, vf, 0, tf)
end

const Optional{T} = Union{T,Missing} where {T}
