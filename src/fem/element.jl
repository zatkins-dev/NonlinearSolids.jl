using LinearAlgebra

export Element, ElementVector, ElementMatrix, Mesh, nodes, element, restrict!, unrestrict!, assemble!, elements, coords, elementindex, dim

struct Element
  nodes::Vector{Int}
  x_loc::AbstractVector
  function Element(nodes, coords)
    new(nodes, @view coords[nodes])
  end
end

Base.length(element::Element) = element.x_loc[end] - element.x_loc[1]
numnodes(element::Element) = length(element.nodes)

struct ElementVector
  element::Ref{Element}
  vector::AbstractVector
  function ElementVector(element)
    vector = zeros(Float64, numnodes(element))
    new(Ref(element), vector)
  end
  function ElementVector(element, vector)
    new(Ref(element), @view vector[element.nodes])
  end
end

struct ElementMatrix
  element::Ref{Element}
  matrix::AbstractMatrix
  function ElementMatrix(element::Element)
    matrix = zeros(Float64, numnodes(element), numnodes(element))
    new(Ref(element), matrix)
  end
  function ElementMatrix(element::Element, matrix::AbstractMatrix)
    new(Ref(element), @view matrix[element.nodes, element.nodes])
  end
end

# for fn in (:length, :size, :axes, :eachindex, :iterate, :first, :last, :view)
#   @eval Base.$fn(evec::ElementVector) = Base.$fn(evec.vector)
#   @eval Base.$fn(emat::ElementMatrix) = Base.$fn(emat.matrix)
# end
Base.length(elvec::ElementVector) = length(elvec.vector)
Base.size(elvec::ElementVector) = size(elvec.vector)
Base.length(elmat::ElementMatrix) = length(elmat.matrix)
Base.size(elmat::ElementMatrix) = size(elmat.matrix)
Base.ndims(::Type{ElementVector}) = 1
Base.ndims(::Type{ElementMatrix}) = 2

element(evec::ElementVector) = evec.element[]
element(emat::ElementMatrix) = emat.element[]

nodes(element::Element) = element.nodes
nodes(evec::ElementVector) = nodes(element(evec))
nodes(emat::ElementMatrix) = nodes(element(emat))

function restrict!(evec::ElementVector, v::AbstractVector)
  """Restrict a vector to the dofs of an elementVector"""
  evec.vector[1:end] .= v[nodes(evec)]
end

function unrestrict!(evec::ElementVector, v::AbstractVector)
  """Unrestrict an elementVector to a global vector"""
  v[nodes(evec)] .+= evec.vector
end

function assemble!(emat::ElementMatrix, K::AbstractMatrix)
  """Assemble the element matrix into the global matrix"""
  K[nodes(emat), nodes(emat)] .+= emat.matrix
end

struct Mesh
  dim::Int
  elements::Vector{Element}
  elements_idx::Dict{Element,Int}
  nodes::Vector{Int}
  coords::AbstractVecOrMat
  function Mesh(dim, elements, nodes, coords)
    element_idx = Dict{Element,Int}(el => i for (i, el) in enumerate(elements))
    new(dim, elements, element_idx, nodes, coords)
  end
end

function elementindex(mesh::Mesh, el::Element)
  get(mesh.elements_idx, el, 0)
end

dim(mesh::Mesh) = mesh.dim
elements(mesh::Mesh) = mesh.elements
nodes(mesh::Mesh) = mesh.nodes
coords(mesh::Mesh) = mesh.coords
