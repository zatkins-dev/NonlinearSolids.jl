export ensurevec, Ramp, LinearInterp

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

struct LinearInterp <: Function
  ts::Vector{Float64}
  vals::AbstractVecOrMat{Float64}
  function LinearInterp(ts::Vector{Float64}, vals::AbstractVecOrMat{Float64})
    if length(ts) != size(vals, 1)
      throw(ArgumentError("Length of ts must match the number of rows in vals"))
    elseif !issorted(ts)
      throw(ArgumentError("ts must be sorted"))
    end
    new(ts, vals)
  end
  function LinearInterp(vals::AbstractVecOrMat{Float64})
    num_ts = size(vals, 1)
    new(LinRange(0, 1, num_ts), vals)
  end
end

function (li::LinearInterp)(t::Number)
  t = Float64(t)
  idx = searchsortedfirst(li.ts, t)
  if idx == 1
    return li.vals[1]
  elseif idx == lastindex(li.ts) + 1
    return li.vals[end]
  else
    t1, t2 = li.ts[idx-1], li.ts[idx]
    v1, v2 = li.vals[idx-1], li.vals[idx]
    return v1 + (v2 - v1) * (t - t1) / (t2 - t1)
  end
end

const Optional{T} = Union{T,Missing} where {T}
