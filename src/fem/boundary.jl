using LinearAlgebra

import Base: values
export Dirichlet, Neumann, TimeDependent, ElementBoundary, nodes, values, apply!, update!, isdirichlet

abstract type AbstractBoundary end
abstract type AbstractElementBoundary <: AbstractBoundary end

nodes(::T) where {T<:AbstractBoundary} = error("nodes not implemented for $(typeof(bc))")
values(::T) where {T<:AbstractBoundary} = error("values not implemented for $(typeof(bc))")
isdirichlet(::T) where {T<:AbstractBoundary} = false
function update!(::T, _) where {T<:AbstractBoundary} end
apply!(bc::AbstractBoundary, K::AbstractMatrix) = apply!(bc, K, missing)
apply!(bc::AbstractBoundary, F::AbstractVector) = apply!(bc, missing, F)
apply!(bc::AbstractBoundary, F::Optional{AbstractVector}, K::Optional{AbstractMatrix}) = apply!(bc, K, F)
apply!(bc::AbstractElementBoundary, K::ElementMatrix) = apply!(bc, K, missing)
apply!(bc::AbstractElementBoundary, F::ElementVector) = apply!(bc, missing, F)
apply!(bc::AbstractBoundary, _, _) = error("apply! not implemented for $(typeof(bc))")

struct Dirichlet <: AbstractBoundary
  nodes::AbstractVector
  values::AbstractVector
end

nodes(bc::Dirichlet) = bc.nodes
values(bc::Dirichlet) = bc.values
isdirichlet(bc::Dirichlet) = true

function apply!(bc::Dirichlet, K::Optional{AbstractMatrix}, F::Optional{AbstractVector})
  @assert ismissing(K)
  if !ismissing(F)
    F[bc.nodes] .= bc.values
  end
end

struct Neumann <: AbstractBoundary
  nodes::AbstractVector
  values::AbstractVector{Float64}
end

nodes(bc::Neumann) = bc.nodes
values(bc::Neumann) = bc.values

function apply!(bc::Neumann, _::Optional{AbstractMatrix}, F::Optional{AbstractVector})
  if !ismissing(F)
    F[bc.nodes] .+= bc.values
  end
end


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
element(bc::ElementBoundary) = bc.element[]

function apply!(bc::ElementBoundary, K::Optional{ElementMatrix}, F::Optional{ElementVector})
  @assert ismissing(K) || element(bc) == element(K)
  @assert ismissing(F) || element(bc) == element(F)
  apply!(bc.bc_el, ismissing(K) ? missing : K.matrix, ismissing(F) ? missing : F.vector)
end

mutable struct TimeDependent{BC<:AbstractBoundary} <: AbstractBoundary
  bc::BC
  update_fn::Function
  pass_context::Bool
end
function TimeDependent(bc::BC, update_fn; pass_context::Bool=false) where {BC<:AbstractBoundary}
  TimeDependent{BC}(bc, update_fn, pass_context)
end

ElementBoundary(el::Element, bc::TimeDependent) = ElementBoundary(el, bc.bc)
isdirichlet(bc::TimeDependent) = isdirichlet(bc.bc)
apply!(bc::TimeDependent, K, F) = apply!(bc.bc, K, F)
nodes(bc::TimeDependent) = nodes(bc.bc)
values(bc::TimeDependent) = values(bc.bc)
function update!(bc::TimeDependent, t, args...; kwargs...)
  if bc.pass_context
    bc.bc.values .= bc.update_fn(bc.bc, t, args...; kwargs...)
  else
    bc.bc.values .= bc.update_fn(t, args...; kwargs...)
  end
end
