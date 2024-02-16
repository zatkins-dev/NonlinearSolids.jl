using LinearAlgebra

export Element, ElementVector, ElementMatrix, Mesh, numdof, nodes, element, restrict!, unrestrict!, assemble!, elements, coords, elementindex, dim

"""Element structure, containing nodes and coordinates."""
struct Element
  nodes::Vector{Int}
  x_loc::AbstractVector
  function Element(nodes, coords)
    new(nodes, @view coords[nodes])
  end
end

"""Length of the 1D element"""
Base.length(element::Element) = element.x_loc[end] - element.x_loc[1]

"""Number of nodes in the element, equal to `p_order - 1`"""
numdof(element::Element) = length(element.nodes)

"""Vector storing field values corresponding to the local nodes of an element."""
struct ElementVector
  element::Ref{Element}
  vector::AbstractVector
  function ElementVector(element)
    vector = zeros(Float64, numdof(element))
    new(Ref(element), vector)
  end
  function ElementVector(element, vector)
    new(Ref(element), @view vector[element.nodes])
  end
end

"""Matrix storing field values corresponding to the local nodes of an element."""
struct ElementMatrix
  element::Ref{Element}
  matrix::AbstractMatrix
  function ElementMatrix(element::Element)
    matrix = zeros(Float64, numdof(element), numdof(element))
    new(Ref(element), matrix)
  end
  function ElementMatrix(element::Element, matrix::AbstractMatrix)
    new(Ref(element), @view matrix[element.nodes, element.nodes])
  end
end

# Redeclare the Base methods for the new types
Base.length(elvec::ElementVector) = length(elvec.vector)
Base.size(elvec::ElementVector) = size(elvec.vector)
Base.length(elmat::ElementMatrix) = length(elmat.matrix)
Base.size(elmat::ElementMatrix) = size(elmat.matrix)
Base.ndims(::Type{ElementVector}) = 1
Base.ndims(::Type{ElementMatrix}) = 2

"""Get underlying element"""
element(evec::ElementVector) = evec.element[]
element(emat::ElementMatrix) = emat.element[]

"""Get global nodes indices corresponding to the local nodes of an element."""
nodes(element::Element) = element.nodes
nodes(evec::ElementVector) = nodes(element(evec))
nodes(emat::ElementMatrix) = nodes(element(emat))

"""Restrict a global vector to the local nodes of an element"""
restrict!(evec::ElementVector, v::AbstractVector) = evec.vector[1:end] .= v[nodes(evec)]

"""Add the local values of an `ElementVector` to a global vector"""
unrestrict!(evec::ElementVector, v::AbstractVector) = v[nodes(evec)] .+= evec.vector

"""Assemble the element matrix into the global matrix"""
assemble!(emat::ElementMatrix, K::AbstractMatrix) = K[nodes(emat), nodes(emat)] .+= emat.matrix

"""Mesh struct, manages elements and global/local nodes"""
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

"""Show the mesh"""
Base.show(io::IO, mesh::Mesh) = println(io, "$(dim(mesh))D mesh with $(length(elements(mesh))) elements and $(length(nodes(mesh))) nodes.")

"""Get the index of an element in the mesh"""
elementindex(mesh::Mesh, el::Element) = get(mesh.elements_idx, el, 0)

"""Get the spatial dimension of the mesh"""
dim(mesh::Mesh) = mesh.dim

"""Get the number of nodes in the mesh"""
numdof(mesh::Mesh) = length(nodes(mesh))

"""Get all elements in the mesh"""
elements(mesh::Mesh) = mesh.elements

"""Get the global nodes of the mesh"""
nodes(mesh::Mesh) = mesh.nodes

"""Get the coordinates of the global nodes"""
coords(mesh::Mesh) = mesh.coords
