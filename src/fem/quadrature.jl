using LinearAlgebra

export integrate, gauss_quadrature, points, weights

abstract type AbstractQuadrature end

points(::AbstractQuadrature) = error("points not implemented for $(typeof(q))")
weights(::AbstractQuadrature) = error("weights not implemented for $(typeof(q))")

struct Quadrature <: AbstractQuadrature
  points::Vector{Float64}
  weights::Vector{Float64}
  function Quadrature(points, weights)
    if length(points) != length(weights)
      error("length(points) != length(weights)")
    end
    new(points, weights)
  end
end

order(q::Quadrature) = length(q.points)
nodes(q::Quadrature) = q.points
weights(q::Quadrature) = q.weights

function gauss_quadrature(order::Int)
  if order == 1
    return Quadrature([0.0], [2.0])
  elseif order == 2
    return Quadrature([-1 / √(3), 1 / √(3)], [1.0, 1.0])
  elseif order == 3
    return Quadrature([-√(3) / √(5), 0.0, √3 / √(5)], [5 / 9, 8 / 9, 5 / 9])
  elseif order == 4
    x1 = √(3 / 7 - 2 * √(6) / (7 * √(5)))
    x2 = √(3 / 7 + 2 * √(6) / (7 * √(5)))
    w1 = (18 + √(30)) / 36
    w2 = (18 - √(30)) / 36
    return Quadrature([-x2, -x1, x1, x2], [w2, w1, w1, w2])
  elseif order == 5
    x1 = 1 / 3 * √(5 - 2 * √(10) / √(7))
    x2 = 1 / 3 * √(5 + 2 * √(10) / √(7))
    w1 = (322 + 13 * √(70)) / 900
    w2 = (322 - 13 * √(70)) / 900
    return Quadrature([-x2, -x1, 0.0, x1, x2], [w2, w1, 128 / 225, w1, w2])
  end
  error("Gauss quadrature not implemented for n = $N")
end

function integrate(q::AbstractQuadrature, f::Function, mode::Symbol=:noindex)
  @assert mode ∈ (:noindex, :index) "Mode must be :noindex or :index"
  if mode == :index
    return integrate(q, [f(ξ, i) for (i, ξ) in enumerate(nodes(q))])
  else
    return integrate(q, [f(ξ) for ξ in nodes(q)])
  end
end

function integrate(q::AbstractQuadrature, f::AbstractVecOrMat)
  return sum(f .* weights(q))
end
