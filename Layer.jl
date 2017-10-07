"""
    Layer{N,M}(activationfn=sigmoid; T=Float64)
    Layer(N, M, activationfn=sigmoid; T=Float64)
    Layer(weights, activationfn=sigmoid)

Constructs a Layer, either by explicitly passing its `weights` or by initializing them at 1 by passing the layer's dimensions (and optionally, a type `T`).
An activation function can also be passed.

A `Layer{N,M}` acts like a ℜᴺ→ℜᴹ function:

# Examples

```jldoctest
julia> layer = Layer{2,3}();

julia> layer([-10, 10])
3-element Array{Float64,1}:
 0.500
 0.500
 0.500

julia> layer(-10, 10)
3-element Array{Float64,1}:
 0.500
 0.500
 0.500
```
"""
struct Layer{N,M}
  weights::AbstractMatrix{Float64}
  activationfn

  function Layer{N,M}(
    weights::AbstractMatrix,
    activationfn=sigmoid) where {N,M}

      if size(weights) != (M,N)
        throw(DimensionMismatch("For a Layer{N,M}, weights should be a (M,N) matrix"))
      end

      if !method_exists(activationfn, (Float64, ))
        throw(ArgumentError("$activationfn is not a valid activation function."))
      end

      return new{N,M}(
        convert(AbstractMatrix{Float64}, weights),
        activationfn)
  end
end

function Layer(
  weights::AbstractMatrix,
  args...)

    return Layer{size(weights)[2],size(weights)[1]}(weights, args...)
end

function Layer{N,M}(args...; T::Type=Float64) where {N,M}

    return Layer(ones(T, (M,N)), args...)
end

function (layer::Layer)(input::AbstractVector)

    return layer.activationfn.(layer.weights * input)
end

function (layer::Layer{N,M})(input::Vararg{Real, N}) where {N,M}

    return layer([input...])
end

function size(::Layer{N,M}) where {N,M}
    return (N,M)
end
