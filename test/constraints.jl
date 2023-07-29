@testset "Constraint builder" begin
    M1 = sparse(3.0I, 4, 4)
    c1 = zeros(4)
    K1 = GeNIOS.ZeroCone(4)
    inds1 = 1:2:7
    con1 = GeNIOS.Constraint(M1, c1, K1, inds1)
    @test length(con1) == 4
    @test size(con1) == (4, 4)

    M, c, K = GeNIOS.build_constraints([con1])
    @test length(c) == 4
    @test length(K) == 4
    @test size(M) == (4, 7)
    @test all(M[CartesianIndex.(1:4, inds1)] .== 3.0)
    @test nnz(M) == 4
    
    M, c, K = GeNIOS.build_constraints([con1], 8)
    @test length(c) == 4
    @test length(K) == 4
    @test size(M) == (4, 8)
    @test all(M[CartesianIndex.(1:4, inds1)] .== 3.0)
    @test nnz(M) == 4
    
    M2 = randn(3, 4)
    inds2 = 2:2:8
    c2 = ones(3)
    K2 = GeNIOS.IntervalCone(ones(3), ones(3))
    con2 = GeNIOS.Constraint(M2, c2, K2, inds2)
    M, c, K = GeNIOS.build_constraints([con1, con2])
    @test nnz(M) == 4 + 3*4
    @test length(c) == 4 + 3
    @test length(K) == 4 + 3
    @test size(M) == (4 + 3, 8)
    @test all(c .== vcat(c1, c2))
    @test all(M[CartesianIndex.(1:4, inds1)] .== 3.0)
    @test all(M[5:7, inds2] .== M2)
    
    M3 = ones(4)'
    inds3 = [3, 4, 5, 6]
    c3 = [1.0]
    K3 = GeNIOS.ZeroCone(1)
    con3 = GeNIOS.Constraint(M3, c3, K3, inds3)
    
    M4 = 2.0I
    inds4 = 1:2
    c4 = [1.0, 2.0]
    K4 = GeNIOS.ZeroCone(2)
    con4 = GeNIOS.Constraint(M4, c4, K4, inds4)
    
    constraints = [con1, con2, con3, con4]
    M, c, K = GeNIOS.build_constraints(constraints, 10)
    @test nnz(M) == 4 + 3*4 + 4 + 2
    @test length(c) == 4 + 3 + 1 + 2
    @test length(K) == 4 + 3 + 1 + 2
    @test size(M) == (4 + 3 + 1 + 2, 10)
    @test all(c .== vcat(c1, c2, c3, c4))
    @test all(M[CartesianIndex.(1:4, inds1)] .== 3.0)
    @test all(M[5:7, inds2] .== M2)
    @test all(M[8, inds3] .== M3)
    @test all(M[9:10, inds4] .== sparse(M4, 2, 2))

end