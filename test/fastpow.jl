using NilPotentIntegrator
using LinearAlgebra
using BenchmarkTools


a = [
0 0 0 1 0
0 0 0 0 1
0 0 0 0 1
0 0 0 0 0
0 0 0 0 0
]

@test_throws "Matrix not nil potent" NilPotentLinearOperator(a + I)
@test_throws "Mutability breaks the caching." NilPotentLinearOperator(a, update_func=identity)

println("Prepairing caching Nilpotent Operator")
@btime NPLO = NilPotentLinearOperator(a)
NPLO = NilPotentLinearOperator(a)

println("Prepairing not-caching Nilpotent Operator")
@btime NPLOn = NilPotentLinearOperator(a; caching=false)
NPLOn = NilPotentLinearOperator(a; caching=false)

@test isapprox(exp(NPLO,80.0),exp(a*80.0))
@test isapprox(exp(NPLOn,80.0),exp(a*80.0))
v = rand(5)
@test isapprox(expmv(NPLO,v,[],80.0),exp(a*80.0)*v)
@test isapprox(expmv(NPLOn,v,[],80.0),exp(a*80.0)*v)
v2 = rand(5)
v3 = rand(5)
expmv!(v2,NPLO,v,[],80.0)
expmv!(v3,NPLOn,v,[],80.0)
@test isapprox(expmv(NPLO,v,[],80.0),v2)
@test isapprox(expmv(NPLO,v,[],80.0),v3)

using BenchmarkTools

println("Speed test, (NilPotent caching, NilPotent not caching, Matrix exp)")
println("Testing exp(t*A)")
@btime exp(NPLO,80.0)
@btime exp(NPLOn,80.0)
@btime exp(a*80.0)
println("Testing exp(t*A)*v")
@btime expmv(NPLO,v,[],80.0)
@btime expmv(NPLOn,v,[],80.0)
@btime exp(a*80.0)*v
println("Comparing inplace vs out of place - caching")
@btime expmv(NPLO,v,[],80.0)
@btime expmv!(v2,NPLO,v,[],80.0) #Out of place faster?
println("Comparing inplace vs out of place - not caching")
@btime expmv(NPLOn,v,[],80.0)
@btime expmv!(v2,NPLOn,v,[],80.0) #Out of place faster?
