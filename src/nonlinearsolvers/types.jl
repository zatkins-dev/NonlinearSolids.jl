module ConvergedReasons
using Revise

export AbstractConvergedReason, REASON_NOT_YET_CONVERGED, REASON_CONVERGED_RELATIVE, REASON_CONVERGED_ABSOLUTE, REASON_DIVERGED_MAXITS, REASON_DIVERGED_STEP, REASON_DIVERGED_LINEAR_SOLVE, is_converged

abstract type AbstractConvergedReason end
struct NotYetConverged <: AbstractConvergedReason end
struct ConvergedReason <: AbstractConvergedReason
  repr::String
end
struct DivergedReason <: AbstractConvergedReason
  repr::String
end

"""Get the string representation of a `AbstractConvergedReason`."""
repr(r::ConvergedReason) = r.repr
repr(r::DivergedReason) = r.repr
repr(::NotYetConverged) = "Not yet converged"
Base.show(io::IO, r::AbstractConvergedReason) = println(io, repr(r))

"""Singleton for a reason of not yet converged."""
const REASON_NOT_YET_CONVERGED = NotYetConverged()
"""Singleton for a reason of converged: relative tolerance."""
const REASON_CONVERGED_RELATIVE = ConvergedReason("Converged: relative tolerance")
"""Singleton for a reason of converged: absolute tolerance."""
const REASON_CONVERGED_ABSOLUTE = ConvergedReason("Converged: absolute tolerance")
"""Singleton for a reason of divergence: maximum iterations."""
const REASON_DIVERGED_MAXITS = DivergedReason("Diverged: maximum iterations")
"""Singleton for a reason of divergence: step tolerance."""
const REASON_DIVERGED_STEP = DivergedReason("Diverged: step tolerance")
"""Singleton for a reason of divergence: linear solve failed."""
const REASON_DIVERGED_LINEAR_SOLVE = DivergedReason("Diverged: linear solve failed")

"""Check if the solver has converged"""
is_converged(::ConvergedReason; allow_maxits::Bool=false) = true
is_converged(r::DivergedReason; allow_maxits::Bool=false) = r === REASON_DIVERGED_MAXITS && allow_maxits
is_converged(::NotYetConverged; allow_maxits::Bool=false) = false
end

import .ConvergedReasons: is_converged
using .ConvergedReasons: AbstractConvergedReason, REASON_CONVERGED_RELATIVE, REASON_CONVERGED_ABSOLUTE, REASON_DIVERGED_LINEAR_SOLVE, REASON_DIVERGED_STEP, REASON_DIVERGED_MAXITS, REASON_NOT_YET_CONVERGED
export is_converged, REASON_CONVERGED_RELATIVE, REASON_CONVERGED_ABSOLUTE, REASON_DIVERGED_LINEAR_SOLVE, REASON_DIVERGED_STEP, REASON_DIVERGED_MAXITS, REASON_NOT_YET_CONVERGED
