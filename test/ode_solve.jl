using NilPotentIntegrator
using OrdinaryDiffEq, Test, DiffEqDevTools
using LinearAlgebra, Random
using BenchmarkTools
using Sundials

# https://github.com/SciML/OrdinaryDiffEq.jl/blob/5a5d88748550f3b69a0d1a89ec3619e0cc0b8417/test/algconvergence/linear_method_tests.jl#L5-L16
Random.seed!(42)
n = 150

M = zeros(n,n)
for i = 1:n-1
    M[i,i+1] = i%3 * 10/i * rand((-1,1))
end
println("Prepairing Nilpotent Operator")
@btime A = NilPotentLinearOperator(M)
A = NilPotentLinearOperator(M)

u0 = ones(n).*Array(1.0:n)
prob = ODEProblem(A,u0,(0.0,100.0))
println("Solving ",n," by ",n," frictionless particle system")
println("Analytic using exponential")
@time sol_analytic = exp(100.0 * Matrix(A)) * u0

println("Nilpotent Integrator krylov=:off")
@time sol1 = solve(prob, LinearExponential(krylov=:off))(100.0)
@test isapprox(sol1, sol_analytic, rtol=1e-10)

println("Nilpotent Integrator krylov=:off fixed timestep 20.0")
@time sol1 = solve(prob, LinearExponential(krylov=:off), adaptive=false, dt = 20.0)(100.0)
@test isapprox(sol1, sol_analytic, rtol=1e-10)


println("Nilpotent Integrator krylov=:off adaptive timestep 20.0")
@time sol1 = solve(prob, LinearExponential(krylov=:off),  dt = 20.0)(100.0)
@test isapprox(sol1, sol_analytic, rtol=1e-10)

println("Nilpotent Integrator krylov=:simple")
@time sol2 = solve(prob, LinearExponential(krylov=:simple))(100.0)
@test isapprox(sol2, sol_analytic, rtol=1e-10)


println("Nilpotent Integrator krylov=:simple fixed timestep 20.0 ")
@time sol2 = solve(prob, LinearExponential(krylov=:simple), adaptive=false, dt = 20.0)(100.0)
@test isapprox(sol2, sol_analytic, rtol=1e-10)

println("Nilpotent Integrator krylov=:simple adaptive timestep 20.0 ")
@time sol2 = solve(prob, LinearExponential(krylov=:simple), dt = 20.0)(100.0)
@test isapprox(sol2, sol_analytic, rtol=1e-10)


#println("Nilpotent Integrator krylov=:adaptive")
#@time sol3 = solve(prob, LinearExponential(krylov=:adaptive))(100.0)
#@test isapprox(sol3, sol_analytic, rtol=1e-10)

#println("Nilpotent Integrator krylov=:adaptive fixed timestep 20.0 ")
#@test_broken @time sol3 = solve(prob, LinearExponential(krylov=:adaptive), adaptive=false, dt = 20.0)(100.0)
#@test isapprox(sol3, sol_analytic, rtol=1e-10)

#println("Nilpotent Integrator krylov=:adaptive adaptive timestep 20.0 ")
#@time sol3 = solve(prob, LinearExponential(krylov=:adaptive), dt = 20.0)(100.0)
#@test isapprox(sol3, sol_analytic, rtol=1e-10)

println("Nilpotent analytic")
@btime sol4 = expmv(A,u0,[],100.0)
sol4 = expmv(A,u0,[],100.0)
@test isapprox(sol4, sol_analytic, rtol=1e-10)


A2 = DiffEqArrayOperator(M)
prob2 = ODEProblem(A2,u0,(0.0,100.0))

println("LinearExponetial Integrator krylov=:off")
@time sol1 = solve(prob2, LinearExponential(krylov=:off))(100.0)

println("LinearExponetial Integrator krylov=:simple")
@time sol2 = solve(prob2, LinearExponential(krylov=:simple))(100.0)

#println("LinearExponetial Integrator krylov=:adaptive")
#@time sol3 = solve(prob2, LinearExponential(krylov=:adaptive))(100.0)

f = (du,u,p,t) -> (du .= M * u)
prob2 = ODEProblem(f,u0,(0.0,100.0))
println("Integrator Tsit5")
@time sol1 = solve(prob2, Tsit5())(100.0)
isapprox(sol1, sol_analytic, rtol=1e-10)

println("Integrator AutoVern7(Rodas5())")
@time sol1 = solve(prob2, AutoVern7(Rodas5()))(100.0)
isapprox(sol1, sol_analytic, rtol=1e-10)

println("Integrator Rosenbrock23")
@time sol1 = solve(prob2, Rosenbrock23())(100.0)
isapprox(sol1, sol_analytic, rtol=1e-10)

println("Integrator BS3")
@time sol1 = solve(prob2, BS3())(100.0)
isapprox(sol1, sol_analytic, rtol=1e-10)

println("Integrator Vern7")
@time sol1 = solve(prob2, Vern7())(100.0)
isapprox(sol1, sol_analytic, rtol=1e-10)

println("Integrator Rodas5")
@time sol1 = solve(prob2, Rodas5())(100.0)
isapprox(sol1, sol_analytic, rtol=1e-10)

println("Integrator KenCarp4")
@time sol1 = solve(prob2, KenCarp4())(100.0)
isapprox(sol1, sol_analytic, rtol=1e-10)

println("Integrator TRBDF2")
@time sol1 = solve(prob2, TRBDF2())(100.0)
isapprox(sol1, sol_analytic, rtol=1e-10)

println("Integrator dopri5")
@time sol1 = solve(prob2, DP5())(100.0)
isapprox(sol1, sol_analytic, rtol=1e-10)

println("Integrator CVODE_BDF")
@time sol1 = solve(prob2, CVODE_BDF())(100.0)
isapprox(sol1, sol_analytic, rtol=1e-10)

println("Integrator CVODE_Adams")
@time sol1 = solve(prob2, CVODE_Adams())(100.0)
isapprox(sol1, sol_analytic, rtol=1e-10)
