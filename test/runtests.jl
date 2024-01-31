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

end
