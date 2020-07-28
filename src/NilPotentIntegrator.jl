module NilPotentIntegrator
using LinearAlgebra
using ArrayInterface

# TODO import vs using?
import DiffEqBase.AbstractDiffEqLinearOperator
import DiffEqBase.DEFAULT_UPDATE_FUNC

export NilPotentLinearOperator, expmv, expmv!

struct NilPotentLinearOperator{T,AType<:AbstractMatrix{T},F,C}  <: AbstractDiffEqLinearOperator{T}
    A::AType
    update_func::F
    cache::C
    index::Int64
    function NilPotentLinearOperator(A::AType;
        update_func=DEFAULT_UPDATE_FUNC, caching=true) where {AType}

        (update_func!=DEFAULT_UPDATE_FUNC) && throw("Mutability breaks the caching.")
        if caching
            n = max(size(A)...)
            cache = Vector{AType}(undef, n+1) #
            sim = deepcopy(A) # TODO necessary?
            i = 1;
            cache[i] = sim
            while !iszero(cache[i])
                tmp = similar(sim);
                mul!(tmp,cache[i],A);
                i += 1;
                (i > n) && throw("Matrix not nil potent") # I trusted you.
                cache[i] = tmp;
            end
            new{eltype(A),AType,typeof(update_func),typeof(cache)}(A, update_func, cache, i)
        else
            n = max(size(A)...)
            sim = deepcopy(A) # TODO necessary?
            sim2 = deepcopy(A)
            i = 1;
            while !iszero(sim2)
                mul!(sim,sim2,A); # That doesn't work
                sim2, sim = sim, sim2
                i += 1;
                (i > n) && throw("Matrix not nil potent") # I trusted you.
            end
            new{eltype(A),AType,typeof(update_func),Bool}(A, update_func, false, i)
        end
    end
end

(L::NilPotentLinearOperator)(u,p,t) = (update_coefficients!(L,u,p,t); expmv(L,u,p,t))
(L::NilPotentLinearOperator)(du,u,p,t) = (update_coefficients!(L,u,p,t); expmv!(du,L,u,p,t))

# Taken without question from: https://github.com/watsona4/dot_julia/blob/36266326353f14d6899db1edd2dd2b685dda392b/packages/DiffEqBase/LCorD/src/operators/diffeq_operator.jl#L14
Base.eltype(L::NilPotentLinearOperator{T,AType,F,C}) where {T,AType,F,C} = T

# Taken without question from: https://github.com/SciML/DiffEqBase.jl/blob/3d94316d2fff0cffd2dfca0050f1f3cc3b11c2dd/src/operators/basic_operators.jl#L87-L113
update_coefficients!(L::NilPotentLinearOperator,u,p,t) = (L.update_func(L.A,u,p,t); L)
setval!(L::NilPotentLinearOperator, A) = (L.A = A; L)
isconstant(L::NilPotentLinearOperator) = L.update_func == DEFAULT_UPDATE_FUNC
Base.similar(L::NilPotentLinearOperator, ::Type{T}, dims::Dims) where T = similar(L.A, T, dims)

# propagate_inbounds here for the getindex fallback
Base.@propagate_inbounds Base.convert(::Type{AbstractMatrix}, L::NilPotentLinearOperator) = L.A
Base.@propagate_inbounds Base.setindex!(L::NilPotentLinearOperator, v, i::Int) = (L.A[i] = v)
Base.@propagate_inbounds Base.setindex!(L::NilPotentLinearOperator, v, I::Vararg{Int, N}) where {N} = (L.A[I...] = v)

Base.eachcol(L::NilPotentLinearOperator) = eachcol(L.A)
Base.eachrow(L::NilPotentLinearOperator) = eachrow(L.A)
Base.length(L::NilPotentLinearOperator) = length(L.A)
Base.iterate(L::NilPotentLinearOperator,args...) = iterate(L.A,args...)
Base.axes(L::NilPotentLinearOperator) = axes(L.A)
Base.IndexStyle(::Type{<:NilPotentLinearOperator{T,AType,F,C}}) where {T,AType,F,C} = Base.IndexStyle(AType)
Base.copyto!(L::NilPotentLinearOperator, rhs) = (copyto!(L.A, rhs); L)
Base.Broadcast.broadcastable(L::NilPotentLinearOperator) = L
Base.ndims(::Type{<:NilPotentLinearOperator{T,AType,F,C}}) where {T,AType,F,C} = ndims(AType)
ArrayInterface.issingular(L::NilPotentLinearOperator) = ArrayInterface.issingular(L.A)



# https://github.com/SciML/DiffEqBase.jl/blob/52bcd26cbbf64a43228286b5eb91d60c2e917d00/src/operators/diffeq_operator.jl#L50-L55


function LinearAlgebra.exp(L::NilPotentLinearOperator{T,AType,F,C},t)  where {T,AType,F,C<:Vector}
    acc = I + (L.A * t)
    f = 1
    tc = t
    @simd for i=2:L.index
        f *= i
        tc *= t
        @inbounds acc .+= tc/(f)*L.cache[i]
    end
    return acc
end#

function expmv(L::NilPotentLinearOperator{T,AType,F,C},u,p,t)  where {T,AType,F,C<:Vector}
    acc = (L.A * t)
    f = 1
    tc = t
    @simd for i=2:L.index
        f *= i
        tc *= t
        @inbounds acc .+= tc/(f)*L.cache[i]
    end
    return acc*u .+ u
end#
function expmv!(v,L::NilPotentLinearOperator{T,AType,F,C},u,p,t)  where {T,AType,F,C<:Vector}
    v .= u
    @simd for i=1:L.index
        @inbounds v .+= t^i/(factorial(i))*L.cache[i]*u
    end
    v
end

function LinearAlgebra.exp(L::NilPotentLinearOperator{T,AType,F,C},t) where {T,AType,F,C<:Bool}
    acc = I + (L.A * t)
    mul = L.A
    f = 1
    tc = t
    @simd for i=2:L.index
        mul = mul * L.A
        f *= i
        tc *= t
        acc .+= tc/(f)*mul
    end
    return acc
end

function expmv(L::NilPotentLinearOperator{T,AType,F,C},u,p,t) where {T,AType,F,C<:Bool}
    acc = (L.A * t)
    acc2 = (L.A * t)
    acc3 = (L.A * t)
    f = 1
    @simd for i=2:L.index
        f *= i
        acc3 .= acc3 * acc2
        acc .+= acc3/f
    end
    return acc*u .+ u
end#
function expmv!(v,L::NilPotentLinearOperator{T,AType,F,C},u,p,t) where {T,AType,F,C<:Bool}
    v .= u
    acc2 = (L.A * t)
    acc3 = (L.A * t)
    v .+= acc2 * u
    @simd for i=2:L.index
        acc3 .= acc3 * acc2
        v .+= acc3*(u/(factorial(i)))
    end
    v
end

end # module
