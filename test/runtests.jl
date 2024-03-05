using NonlinearSolids
using Test

@testset "NonlinearSolids.jl" begin
  @testset "NewtonResult" begin
    @testset "constructor" begin
      result = NewtonResult(3, numsteps=5, maxits=10)
      @test size(result.d) == (5, 3)
      @test size(result.dₙᵏ) == (5, 10, 3)
      @test size(result.res_d) == (5, 10)
      @test size(result.num_its) == (5,)
    end
    @testset "trim!" begin
      result = NewtonResult(3, numsteps=5, maxits=10)
      result.num_its[2] = 7
      trim!(result)
      @test size(result.d) == (5, 3)
      @test size(result.dₙᵏ) == (5, 7, 3)
      @test size(result.res_d) == (5, 7)
      @test size(result.num_its) == (5,)
      trim!(result, 3)
      @test size(result.d) == (3, 3)
      @test size(result.dₙᵏ) == (3, 7, 3)
      @test size(result.res_d) == (3, 7)
      @test size(result.num_its) == (3,)
    end
  end
  @testset "ArcLengthResult" begin
    @testset "constructor" begin
      result = ArcLengthResult(3, maxsteps=5, maxits=10)
      @test size(result.d) == (5, 3)
      @test size(result.λ) == (5,)
      @test size(result.dₙᵏ) == (5, 10, 3)
      @test size(result.λₙᵏ) == (5, 10)
      @test size(result.res_d) == (5, 10)
      @test size(result.res_λ) == (5, 10)
      @test size(result.num_its) == (5,)
    end
    @testset "trim!" begin
      result = ArcLengthResult(3, maxsteps=5, maxits=10)
      result.num_its[2] = 7
      result.num_steps = 3
      trim!(result)
      @test size(result.d) == (3, 3)
      @test size(result.dₙᵏ) == (3, 7, 3)
      @test size(result.λ) == (3,)
      @test size(result.λₙᵏ) == (3, 7)
      @test size(result.res_d) == (3, 7)
      @test size(result.res_λ) == (3, 7)
      @test size(result.num_its) == (3,)
    end
  end

  @testset "ShapeFunctions" begin
    points = range(-1, 1; length=10001)
    fns_order_1(ξ) = 0.5 * [1 - ξ, 1 + ξ]
    dfns_order_1(ξ) = 0.5 * [-1, 1]
    fns_order_2(ξ) = [-0.5 * ξ * (1 - ξ), 1 - ξ^2, 0.5 * ξ * (1 + ξ)]
    dfns_order_2(ξ) = [ξ - 0.5, -2ξ, ξ + 0.5]

    @testset "Lagrange - order $(p)" for p in 1:3
      @testset "Chebyshev 2nd Type" begin
        shape = lagrange(p, :chebyshev2)
        @test order(shape) == p
        @test nodes(shape) ≈ [cos(i * π / p) for i in p:-1:0]
        @test sum.(N.(Ref(shape), points)) ≈ ones(length(points))
        if p == 1
          @test N.(Ref(shape), points) ≈ fns_order_1.(points)
          @test ∇N.(Ref(shape), points) ≈ dfns_order_1.(points)
        elseif p == 2
          @test N.(Ref(shape), points) ≈ fns_order_2.(points)
          @test ∇N.(Ref(shape), points) ≈ dfns_order_2.(points)
        end
      end
      @testset "Equidistant" begin
        fns_order_3(ξ) = 9 / 16 * [
          -(1 - ξ) * (1 / 3 + ξ) * (1 / 3 - ξ),
          3 * (1 - ξ) * (1 + ξ) * (1 / 3 - ξ),
          3 * (1 - ξ) * (1 + ξ) * (1 / 3 + ξ),
          -(1 + ξ) * (1 / 3 + ξ) * (1 / 3 - ξ)
        ]
        shape = lagrange(p, :equidistant)
        @test order(shape) == p
        @test nodes(shape) ≈ LinRange(-1, 1, p + 1)
        @test sum.(N.(Ref(shape), points)) ≈ ones(length(points))
        if p == 1
          @test N.(Ref(shape), points) ≈ fns_order_1.(points)
          @test ∇N.(Ref(shape), points) ≈ dfns_order_1.(points)
        elseif p == 2
          @test N.(Ref(shape), points) ≈ fns_order_2.(points)
          @test ∇N.(Ref(shape), points) ≈ dfns_order_2.(points)
        elseif p == 3
          @test N.(Ref(shape), points) ≈ fns_order_3.(points)
        end
      end
    end
  end

  @testset "Quadrature" begin
    @testset "Gauss Quadrature" begin
      @testset "integrate exact" begin
        # integrate polynomial with order 2N-1 exactly
        q = gauss_quadrature(1)
        f1(ξ) = ξ - 1
        @test integrate(q, f1) ≈ -2
        q = gauss_quadrature(2)
        f2(ξ) = ξ^3 - 2ξ^2 + 3ξ - 4
        @test integrate(q, f2) ≈ -28 / 3
        q = gauss_quadrature(3)
        f3(ξ) = ξ^5 - 2ξ^4 + 3ξ^3 - 4ξ^2 + 5ξ - 6
        @test integrate(q, f3) ≈ -232 / 15
        q = gauss_quadrature(4)
        f4(ξ) = ξ^7 - 2ξ^6 + 3ξ^5 - 4ξ^4 + 5ξ^3 - 6ξ^2 + 7ξ - 8
        @test integrate(q, f4) ≈ -776 / 35
        q = gauss_quadrature(5)
        f5(ξ) = ξ^9 - 2ξ^8 + 3ξ^7 - 4ξ^6 + 5ξ^5 - 6ξ^4 + 7ξ^3 - 8ξ^2 + 9ξ - 10
        @test integrate(q, f5) ≈ -9236 / 315
      end
      @testset "integrate nonlinear" begin
        f1(ξ) = tanh(1 + ξ)
        q = gauss_quadrature(3)
        @test integrate(q, f1) ≈ -2 - log(2) + log(1 + exp(4)) rtol = 5e-4
        q = gauss_quadrature(4)
        @test integrate(q, f1) ≈ -2 - log(2) + log(1 + exp(4)) rtol = 5e-5
        q = gauss_quadrature(5)
        @test integrate(q, f1) ≈ -2 - log(2) + log(1 + exp(4)) rtol = 5e-6
        f2(ξ) = sin(exp(ξ^2))
        f2_exact = 1.77247907969601871352278360619085214927452859669568635245
        q = gauss_quadrature(3)
        @test integrate(q, f2) ≈ f2_exact rtol = 5e-2
        q = gauss_quadrature(4)
        @test integrate(q, f2) ≈ f2_exact rtol = 5e-3
        q = gauss_quadrature(5)
        @test integrate(q, f2) ≈ f2_exact rtol = 6e-4
      end
    end
  end

  @testset "Boundaries" begin
    @testset "Dirichlet" begin
      bc = Dirichlet([1, 3], [1.0, 3.0])
      vec = zeros(4)
      apply!(bc, vec)
      @test vec ≈ [1.0, 0.0, 3.0, 0.0]
    end
    @testset "Dirichlet Element" begin
      @testset "Test apply" begin
        el = Element([2, 3], [0.0, 0.5, 1.0])
        bc = Dirichlet([1, 3], [1.0, 3.0])
        el_bc = ElementBoundary(el, bc)
        @test nodes(el_bc) ≈ [2]
        @test values(el_bc) ≈ [3.0]
        @test element(el_bc) === el
        vec = ElementVector(el)
        apply!(el_bc, vec)
        @test vec.vector ≈ [0.0, 3.0]
      end
      @testset "Test num BC nodes < num Element nodes" begin
        el = Element([2, 3], [0.0, 0.5, 1.0])
        bc = Dirichlet([3], [3.0])
        el_bc = ElementBoundary(el, bc)
        @test nodes(el_bc) ≈ [2]
        @test values(el_bc) ≈ [3.0]
        @test element(el_bc) === el
        vec = ElementVector(el)
        apply!(el_bc, vec)
        @test vec.vector ≈ [0.0, 3.0]
      end
    end
    @testset "Time Dependent" begin
      @testset "Dirichlet" begin
        update_fn(_, t) = [1.0 * t, 3.0 * t]
        bc = TimeDependent(Dirichlet([1, 3], [0.0, 0.0]), update_fn, pass_context=true)
        vec = zeros(4)
        apply!(bc, vec)
        @test vec ≈ [0.0, 0.0, 0.0, 0.0]
        update!(bc, 0.5)
        apply!(bc, vec)
        @test vec ≈ [0.5, 0.0, 1.5, 0.0]
        update!(bc, 1.0)
        apply!(bc, vec)
        @test vec ≈ [1.0, 0.0, 3.0, 0.0]
      end
      @testset "Dirichlet Element" begin
        el = Element([2, 3], [0.0, 1 / 3, 2 / 3, 1.0])
        update_fn(t) = [1.0 * t, 3.0 * t]
        bc = TimeDependent(Dirichlet([1, 3], [0.0, 0.0]), update_fn)
        el_bc = ElementBoundary(el, bc)
        @test nodes(el_bc) ≈ [2]
        @test values(el_bc) ≈ [0.0]
        @test element(el_bc) === el
        # Test initial apply
        vec = ElementVector(el)
        apply!(el_bc, vec)
        @test vec.vector ≈ [0.0, 0.0]
        # Test update
        vec.vector .= zeros(2)
        update!(bc, 0.5)
        apply!(el_bc, vec)
        @test vec.vector ≈ [0.0, 1.5]
        # Test update
        update!(bc, 1.0)
        apply!(el_bc, vec)
        @test vec.vector ≈ [0.0, 3.0]
      end
    end
  end
end
