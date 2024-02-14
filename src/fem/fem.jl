using LinearAlgebra

export FEMModel, getfemfield!, getfemfield_el!, addboundary!, updateboundaries!, applydirichletboundaries!, applyneumannboundaries!, step!, dofs, expand, addpostprocess!, postprocess!, gettime, getstep

mutable struct FEMModel
  mesh::Mesh
  Q::AbstractQuadrature
  P::AbstractShapeFunctions
  fields::Dict
  fields_el::Dict
  boundaries::Vector{AbstractBoundary}
  boundaries_el::Vector{Vector{AbstractElementBoundary}}
  data::Dict
  time::Float64
  step::Int
  constrained_nodes::BitVector
  postprocess::Vector{Function}
  function FEMModel(mesh::Mesh, q, p; data=Dict())
    Q = gauss_quadrature(Val(q))
    P = lagrange(p, :chebyshev2)
    boundaries_el = [AbstractElementBoundary[] for _ in elements(mesh)]
    constrained_nodes = falses(length(nodes(mesh)))
    new(mesh, Q, P, Dict(), Dict(), AbstractBoundary[], boundaries_el, data, 0.0, 1, constrained_nodes, [])
  end
end

numdof(fem::FEMModel) = length(nodes(fem.mesh))
dim(fem::FEMModel) = dim(fem.mesh)

function getfemfield_el!(fem::FEMModel, field::Symbol; ismat::Bool=false)
  if !haskey(fem.fields_el, field)
    fem.fields_el[field] = [ismat ? ElementMatrix(elem) : ElementVector(elem) for elem in elements(fem.mesh)]
  end
  return fem.fields_el[field]
end

function getfemfield!(fem::FEMModel, field::Symbol; dims=missing)
  if ismissing(dims)
    dims = (dim(mesh),)
  end
  if !haskey(fem.fields, field)
    fem.fields[field] = zeros(Float64, (length(nodes(fem.mesh)), dims...))
  end
  return fem.fields[field]
end

function addboundary!(fem::FEMModel, bc::AbstractBoundary)
  push!(fem.boundaries, bc)
  for (el, boundaries) in zip(elements(fem.mesh), fem.boundaries_el)
    push!(boundaries, ElementBoundary(el, bc))
  end
  if isdirichlet(bc)
    fem.constrained_nodes[nodes(bc)] .= 1
  end
end

"""
Adds a postprocessing function to the model

Signature: f(fem::FEMModel, u::AbstractVector, res::NewtonFEMResult)

The function should modify the res argument in place.
"""
function addpostprocess!(fem::FEMModel, f::Function)
  push!(fem.postprocess, f)
end

function postprocess!(fem::FEMModel, u::AbstractVector, res::AbstractSolverResult)
  for f in fem.postprocess
    f(fem, u, res)
  end
end

function gettime(fem::FEMModel)
  fem.time
end

function getstep(fem::FEMModel)
  fem.step
end

function step!(fem::FEMModel, dt)
  fem.time += dt
  fem.step += 1
  updateboundaries!(fem)
end

function updateboundaries!(fem::FEMModel)
  for bc in fem.boundaries
    update!(bc, fem.time)
  end
end

function applydirichletboundaries!(fem::FEMModel, el::Element, args...)
  el_idx = elementindex(fem.mesh, el)
  for bc in fem.boundaries_el[el_idx]
    if isdirichlet(bc)
      apply!(bc, args...)
    end
  end
end

function applydirichletboundaries!(fem::FEMModel, v)
  for bc in fem.boundaries
    if isdirichlet(bc)
      apply!(bc, v)
    end
  end
end

function applyneumannboundaries!(fem::FEMModel, v)
  for bc in fem.boundaries
    if !isdirichlet(bc)
      apply!(bc, v)
    end
  end
end

dofs(fem::FEMModel, v::AbstractVector) = @view v[.~fem.constrained_nodes]
dofs(fem::FEMModel, A::AbstractMatrix) = @view A[.~fem.constrained_nodes, .~fem.constrained_nodes]
expand(fem::FEMModel, v::AbstractVector) = begin
  newv = zeros(length(fem.constrained_nodes))
  newv[.~fem.constrained_nodes] .= v
  newv
end
expand(fem::FEMModel, A::AbstractMatrix) = begin
  newA = zeros(length(fem.constrained_nodes), length(fem.constrained_nodes))
  newA[.~fem.constrained_nodes, .~fem.constrained_nodes] .= A
  newA
end
