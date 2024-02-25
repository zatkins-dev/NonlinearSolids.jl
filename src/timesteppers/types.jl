export AbstractTimeStepper, solver, model, numdof, maxsteps, time, getoutputarray!, trim!, addpostprocess!, postprocess!, step!
import Base: time
abstract type AbstractTimeStepper end

solver(ts::AbstractTimeStepper) = ts.solver
model(ts::AbstractTimeStepper) = model(solver(ts))
numdof(ts::AbstractTimeStepper) = numdof(solver(ts))
maxsteps(ts::AbstractTimeStepper) = ts.maxsteps
"""Get the current time from the timestepper"""
time(ts::AbstractTimeStepper) = ts.t
"""Get the current step from the timestepper"""
Base.step(ts::AbstractTimeStepper) = ts.step

"""Get or create an array of size `(maxsteps, dims...)` to store output data"""
function getoutputarray!(ts::AbstractTimeStepper, key::Symbol, dims=missing)
  if ismissing(dims)
    dims = (maxsteps(ts), numdof(ts),)
  elseif Base.haslength(dims)
    dims = (maxsteps(ts), dims...)
  else
    dims = (maxsteps(ts), dims)
  end
  if !haskey(ts.output_data, key)
    ts.output_data[key] = zeros(Float64, dims)
  end
  return ts.output_data[key]
end

function Base.getproperty(ts::T, key::Symbol) where {T<:AbstractTimeStepper}
  if hasfield(T, key)
    return getfield(ts, key)
  elseif hasfield(T, :output_data) && haskey(ts.output_data, key)
    return getoutputarray!(ts, key)
  else
    return getfield(ts, key)
  end
end

function trim!(ts::T) where {T<:AbstractTimeStepper}
  if !hasfield(T, :output_data)
    return
  end
  for (key, val) in result.output_data
    result.output_data[key] = val[1:step(ts), :]
  end
end

"""
Adds a postprocessing function to the timestepper

Signature: f(ts::T, u::AbstractVector)

The function should modify the ts argument in place using getoutputarray!.
"""
function addpostprocess!(ts::AbstractTimeStepper, f::Function)
  push!(ts.postprocess, f)
end

"""Apply all postprocessing functions"""
function postprocess!(ts::AbstractTimeStepper, u::AbstractVector)
  for f in ts.postprocess
    f(ts, u)
  end
end

function step!(ts::AbstractTimeStepper, Δt::Real)
  ts.t += Δt
  ts.step += 1
  updateboundaries!(model(ts), time(ts))
end
