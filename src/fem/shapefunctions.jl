using LinearAlgebra

export AbstractShapeFunctions, lagrange, Lagrange, N, ∇N, order, interpolate

abstract type AbstractShapeFunctions end

order(::T) where {T<:AbstractShapeFunctions} = error("order not implemented for $(T)")

struct Lagrange <: AbstractShapeFunctions
  p::Int
  w::Vector{Float64}
  nodes::Vector{Float64}
end

function order(l::Lagrange)
  return l.p
end

nodes(shape::Lagrange) = shape.nodes

function lagrange(order::Int, nodes)
  if length(nodes) != order + 1
    error("length(nodes) != order+1")
  end
  w = ones(order + 1)
  for i in eachindex(nodes)
    w[i] = prod(nodes[i] .- nodes[1:i-1]) * prod(nodes[i] .- nodes[i+1:end])
  end
  w = 1 ./ w
  return Lagrange(order, w, nodes)
end

function lagrange(order::Int, t::Symbol)
  if t == :equidistant
    nodes = LinRange(-1, 1, order + 1)
    wj = (-1) .^ (order:-1:0) .* [binomial(order, i) for i in order:-1:0]
    return lagrange(order, nodes)
  elseif t == :chebyshev2
    nodes = [cos(i * π / order) for i in order:-1:0]
    return lagrange(order, nodes)
  end
  error("Unknown node type $t")
end

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

function interpolate(shape::Lagrange, f::Function, ξ::Number)
  return dot(f.(shape.nodes), N(shape, ξ))
end

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
