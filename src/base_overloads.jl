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
