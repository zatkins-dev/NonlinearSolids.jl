using LinearAlgebra

export AbstractShapeFunctions, lagrange, Lagrange, N, ∇N, order, nodes, weights, interpolate

abstract type AbstractShapeFunctions end

"""Lagrange interpolating polynomial shape functions."""
struct Lagrange <: AbstractShapeFunctions
  w::Vector{Float64}
  nodes::Vector{Float64}
end

"""Get the order of the shape functions."""
order(shape::Lagrange) = length(nodes(shape)) - 1

"""Get the nodes of the shape functions."""
nodes(shape::Lagrange) = shape.nodes

"""Get the weights of the shape functions."""
weights(shape::Lagrange) = shape.w

"""Pre-compute the weights for the Lagrange interpolating polynomials at the given points."""
function lagrange(points)
  w = ones(length(points))
  for i in eachindex(points)
    w[i] = prod(points[i] .- points[1:i-1]) * prod(points[i] .- points[i+1:end])
  end
  w = 1 ./ w
  return Lagrange(w, points)
end

"""Create Lagrange shape functions of given order and node type.

Supported node types:
  - `:equidistant`: Equidistant nodes in the interval `[-1, 1]`
  - `:chebyshev2`: Chebyshev nodes of the second kind
"""
function lagrange(order::Int, t::Symbol)
  if t == :equidistant
    nodes = LinRange(-1, 1, order + 1)
    return lagrange(nodes)
  elseif t == :chebyshev2
    nodes = [cos(i * π / order) for i in order:-1:0]
    return lagrange(nodes)
  end
  error("Unknown node type $t")
end

"""Get the value of the shape functions at the given point."""
function N(shape::Lagrange, ξ::Number)
  is_node_arr = ξ .≈ shape.nodes
  if any(is_node_arr)
    return vec(Float64.(is_node_arr))
  end
  ℓ = prod(ξ .- shape.nodes)
  ℓj = ℓ * ones(length(shape.nodes))
  for i in eachindex(shape.nodes)
    ℓj[i] *= shape.w[i] / (ξ - shape.nodes[i])
  end
  return ℓj
end

"""Get the gradient of the shape functions with respect to ξ at the given point."""
function ∇N(shape::Lagrange, ξ::Number)
  dℓj = zeros(length(shape.nodes))
  is_node_arr = ξ .≈ shape.nodes
  idx = findfirst(is_node_arr)
  if !isnothing(idx)
    for j in eachindex(shape.nodes)
      if j != idx
        dℓj[j] = (shape.w[j] / shape.w[idx]) / (shape.nodes[idx] - shape.nodes[j])
      end
    end
    dℓj[idx] = -sum(dℓj)
    return dℓj
  end
  for i in eachindex(shape.nodes)
    dℓj[i] = sum(1 / (ξ - shape.nodes[j]) for j in eachindex(shape.nodes) if j != i)
  end
  return N(shape, ξ) .* dℓj
end

"""Interpolate a function using the shape functions.

The signature of the function is `f(ξ)`.
"""
function interpolate(shape::Lagrange, f::Function, ξ::Number)
  return dot(f.(nodes(shape)), N(shape, ξ))
end

"""Interpolate the function values using the shape functions."""
function interpolate(shape::Lagrange, f::AbstractVector, ξ::Number)
  return dot(f, N(shape, ξ))
end
