export DistributedForce

struct DistributedForce <: Function
  fdist_fn::Union{Function,Number}
end

function _distributedforce_el_fn(fem::FEMModel, e::Element, fdist)
  J = length(e) / 2 # dx/dξ
  qfunc = (ξ) -> J * fdist * N(fem.P, ξ)
  return integrate(fem.Q, qfunc)
end

function (f::DistributedForce)(fem::FEMModel, U, Fext, ctx; mode=:add)
  if !(mode in (:add, :set))
    throw(ArgumentError("Invalid mode: $mode"))
  end
  fdist_fn = f.fdist_fn
  fdist = fdist_fn isa Function ? fdist_fn(time(ctx)) : fdist_fn
  if mode == :set
    Fext .= 0
  end
  f_f = getfemfield_el!(fem, :f_f)
  for f_f_el in f_f
    f_f_el.vector .= _distributedforce_el_fn(fem, element(f_f_el), fdist)
    unrestrict!(f_f_el, Fext)
  end
end
