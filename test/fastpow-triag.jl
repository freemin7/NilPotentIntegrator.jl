using NilPotentIntegrator
using LinearAlgebra

const a = UpperTriangular([
0 0 0 1 0
0 0 0 0 1
0 0 0 0 1
0 0 0 0 0
0 0 0 0 0
])

const NPLO = NilPotentLinearOperator(a)


@test isapprox(exp(NPLO,80.0),exp(a*80.0))
v = rand(5)
@test isapprox(expmv(NPLO,v,[],80.0),exp(a*80.0)*v)
v2 = rand(5)
expmv!(v2,NPLO,v,[],80.0)
@test_broken isapprox(expmv(NPLO,v,[],80.0),v2) # out of place doesn't work?

using BenchmarkTools

println("Speed test, UpperTriangular version , NilPotent version first")
println("Testing exp(t*A)")
@btime exp(NPLO,80.0)
@btime exp(a*80.0)
println("Testing exp(t*A)*v")
@btime expmv(NPLO,v,[],80.0)
@btime exp(a*80.0)*v
println("Comparing inplace vs out of place")
@btime expmv(NPLO,v,[],80.0)
@btime expmv!(v2,NPLO,v,[],80.0) #Out of place faster?
