using LinearAlgebra

export FEMModel, getfemfield_el!, addboundary!, updateboundaries!, applydirichletboundaries!, applyneumannboundaries!, step!, dofs, expand, addpostprocess!, postprocess!, gettime, getstep

"""Finite Element Model

Consolidates the mesh, quadrature, shape functions, fields, boundaries, and user-provided data.
"""
mutable struct FEMModel
  mesh::Mesh
  Q::AbstractQuadrature
  P::AbstractShapeFunctions
  fields_el::Dict
  boundaries::Vector{AbstractBoundary}
  boundaries_el::Vector{Vector{AbstractElementBoundary}}
  data::Dict
  time::Float64
  step::Int
  constrained_nodes::BitVector
  postprocess::Vector{Function}
  function FEMModel(mesh::Mesh, q, p; data=Dict())
    Q = gauss_quadrature(q)
    P = lagrange(p, :chebyshev2)
    boundaries_el = [AbstractElementBoundary[] for _ in elements(mesh)]
    constrained_nodes = falses(length(nodes(mesh)))
    new(mesh, Q, P, Dict(), AbstractBoundary[], boundaries_el, data, 0.0, 1, constrained_nodes, [])
  end
end

"""Show the model"""
function Base.show(io::IO, fem::FEMModel)
  print(io, "FEMModel\n  ", fem.mesh, "  Quadrature: Gauss, order $(order(fem.Q))\n  Shape functions: Lagrange, order $(order(fem.P))\n  Fields: ", keys(fem.fields_el), "\n  Boundaries: ", fem.boundaries, "\n  Postprocess: ", fem.postprocess)
end

"""Get the number of degrees of freedom in the model"""
numdof(fem::FEMModel) = length(nodes(fem.mesh))

"""Get the spatial dimension of the model"""
dim(fem::FEMModel) = dim(fem.mesh)

"""Get a list of `ElementVector` for the given field name"""
function getfemfield_el!(fem::FEMModel, field::Symbol; ismat::Bool=false)
  if !haskey(fem.fields_el, field)
    fem.fields_el[field] = [ismat ? ElementMatrix(elem) : ElementVector(elem) for elem in elements(fem.mesh)]
  end
  return fem.fields_el[field]
end

"""Add a boundary condition to the model"""
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

"""Apply all postprocessing functions"""
function postprocess!(fem::FEMModel, u::AbstractVector, res::AbstractSolverResult)
  for f in fem.postprocess
    f(fem, u, res)
  end
end

"""Get the current time"""
function gettime(fem::FEMModel)
  fem.time
end

"""Get the current step"""
function getstep(fem::FEMModel)
  fem.step
end

"""Advance the model by `dt`, increase the step count, and update BCs"""
function step!(fem::FEMModel, dt)
  fem.time += dt
  fem.step += 1
  updateboundaries!(fem)
end

"""Update all time dependent boundary conditions"""
function updateboundaries!(fem::FEMModel)
  for bc in fem.boundaries
    update!(bc, fem.time)
  end
end

"""Apply all Dirichlet boundary conditions to the solution vector"""
function applydirichletboundaries!(fem::FEMModel, v)
  for bc in fem.boundaries
    if isdirichlet(bc)
      apply!(bc, v)
    end
  end
end

"""Apply all Neumann boundary conditions to the right hand side/external force vector"""
function applyneumannboundaries!(fem::FEMModel, v)
  for bc in fem.boundaries
    if !isdirichlet(bc)
      apply!(bc, v)
    end
  end
end

"""Get the degrees of freedom corresponding to the unconstrained nodes of the vector `v`"""
dofs(fem::FEMModel, v::AbstractVector) = @view v[.~fem.constrained_nodes]
"""Get the degrees of freedom corresponding to the unconstrained nodes of the matrix `A`"""
dofs(fem::FEMModel, A::AbstractMatrix) = @view A[.~fem.constrained_nodes, .~fem.constrained_nodes]

"""Zero-pad a vector to include constrained nodes"""
expand(fem::FEMModel, v::AbstractVector) = begin
  newv = zeros(length(fem.constrained_nodes))
  newv[.~fem.constrained_nodes] .= v
  newv
end
"""Zero-pad a matrix to include constrained nodes"""
expand(fem::FEMModel, A::AbstractMatrix) = begin
  newA = zeros(length(fem.constrained_nodes), length(fem.constrained_nodes))
  newA[.~fem.constrained_nodes, .~fem.constrained_nodes] .= A
  newA
end
