using NilPotentIntegrator
using OrdinaryDiffEq, Test, DiffEqDevTools
using LinearAlgebra, Random
using BenchmarkTools

# https://github.com/SciML/OrdinaryDiffEq.jl/blob/5a5d88748550f3b69a0d1a89ec3619e0cc0b8417/test/algconvergence/linear_method_tests.jl#L5-L16

n = 150

M = zeros(n,n)
for i = 1:n-1
    M[i,i+1] = i%3 * 10/n * rand((-1,1))
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

println("Nilpotent Integrator krylov=:simple")
@time sol2 = solve(prob, LinearExponential(krylov=:simple))(100.0)
@test isapprox(sol2, sol_analytic, rtol=1e-10)

println("Nilpotent analytic")
@btime sol4 = expmv(A,u0,[],100.0)
sol4 = expmv(A,u0,[],100.0)
@test isapprox(sol4, sol_analytic, rtol=1e-10)

#sol3 = solve(prob, LinearExponential(krylov=:adaptive))(40.0)
#@test isapprox(sol3, sol_analytic, rtol=1e-10)


A2 = DiffEqArrayOperator(M)
prob2 = ODEProblem(A2,u0,(0.0,100.0))

println("LinearExponetial Integrator krylov=:off")
@time sol1 = solve(prob2, LinearExponential(krylov=:off))(100.0)

println("LinearExponetial Integrator krylov=:simple")
@time sol2 = solve(prob2, LinearExponential(krylov=:simple))(100.0)

#println("LinearExponetial Integrator krylov=:adaptive")
#@time sol3 = solve(prob2, LinearExponential(krylov=:adaptive))(100.0)
