using SafeTestsets
using Test

const GROUP = get(ENV, "GROUP", "All")

@time begin
if GROUP == "All" || GROUP == "Core"
    @time @safetestset "Fast Power" begin include("fastpow.jl") end
    @time @safetestset "ode solve large" begin include("ode_solve.jl") end
    #@time @safetestset "Fast Triangular Power" begin include("fastpow-triag.jl") end
end
end
