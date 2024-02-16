using LinearAlgebra

import Base: values
export Dirichlet, Neumann, TimeDependent, ElementBoundary, nodes, values, apply!, update!, isdirichlet

abstract type AbstractBoundary end
abstract type AbstractElementBoundary <: AbstractBoundary end

"""Get the nodes from a boundary condition."""
nodes(::T) where {T<:AbstractBoundary} = error("nodes not implemented for $(typeof(bc))")
"""Get the values at nodes from a boundary condition."""
values(::T) where {T<:AbstractBoundary} = error("values not implemented for $(typeof(bc))")
"""Apply boundary condition to a vector."""
apply!(bc::AbstractBoundary, _) = error("apply! not implemented for $(typeof(bc))")
"""Check if boundary condition is Dirichlet."""
isdirichlet(::T) where {T<:AbstractBoundary} = false
"""Update boundary condition values, only affects `TimeDependent` boundaries."""
function update!(::T, _) where {T<:AbstractBoundary} end

"""Dirichlet/Essential boundary condition, prescribes solution values at nodes."""
struct Dirichlet <: AbstractBoundary
  nodes::AbstractVector
  values::AbstractVector
end
nodes(bc::Dirichlet) = bc.nodes
values(bc::Dirichlet) = bc.values
isdirichlet(bc::Dirichlet) = true
"""Apply Dirichlet boundary condition to the solution vector."""
apply!(bc::Dirichlet, D::AbstractVector) = D[bc.nodes] .= bc.values

"""Neumann/Natural boundary condition, prescribes fluxes at nodes."""
struct Neumann <: AbstractBoundary
  nodes::AbstractVector
  values::AbstractVector{Float64}
end
nodes(bc::Neumann) = bc.nodes
values(bc::Neumann) = bc.values
"""Apply Neumann boundary condition to the right hand side/external force vector."""
apply!(bc::Neumann, F::AbstractVector) = F[bc.nodes] .+= bc.values

"""Boundary condition for a single element."""
struct ElementBoundary <: AbstractElementBoundary
  element::Ref{Element}
  bc_el::AbstractBoundary
  function ElementBoundary(el::Element, bc::T) where {T<:Union{Dirichlet,Neumann}}
    idx_local = nodes(el) .∈ Ref(nodes(bc))
    idx_vals = nodes(bc) .∈ Ref(nodes(el))
    @views begin
      nodes_local = findall(!iszero, idx_local)
      bc_local = T(nodes_local, values(bc)[idx_vals])
    end
    new(Ref(el), bc_local)
  end
end
isdirichlet(bc::ElementBoundary) = isdirichlet(bc.bc_el)
nodes(bc::ElementBoundary) = nodes(bc.bc_el)
values(bc::ElementBoundary) = values(bc.bc_el)
"""Get the element acted upon by a boundary condition."""
element(bc::ElementBoundary) = bc.element[]
"""Apply boundary condition to an `ElementVector`."""
apply!(bc::ElementBoundary, evec::ElementVector) = apply!(bc.bc_el, evec.vector)

"""Time dependent boundary condition, updates values based on a function of time."""
mutable struct TimeDependent{BC<:AbstractBoundary} <: AbstractBoundary
  bc::BC
  update_fn::Function
  pass_context::Bool
end
"""Create a time dependent boundary condition.
  
Signature of `update_fn`:
  - `update_fn(bc::BC, t, args...; kwargs...)` if `pass_context` is true
  - `update_fn(t, args...; kwargs...)` otherwise
"""
function TimeDependent(bc::BC, update_fn; pass_context::Bool=false) where {BC<:AbstractBoundary}
  TimeDependent{BC}(bc, update_fn, pass_context)
end

ElementBoundary(el::Element, bc::TimeDependent) = ElementBoundary(el, bc.bc)
isdirichlet(bc::TimeDependent) = isdirichlet(bc.bc)
apply!(bc::TimeDependent, v) = apply!(bc.bc, v)
nodes(bc::TimeDependent) = nodes(bc.bc)
values(bc::TimeDependent) = values(bc.bc)
"""Update the boundary condition values using user provided function."""
function update!(bc::TimeDependent, t, args...; kwargs...)
  if bc.pass_context
    bc.bc.values .= bc.update_fn(bc.bc, t, args...; kwargs...)
  else
    bc.bc.values .= bc.update_fn(t, args...; kwargs...)
  end
end
