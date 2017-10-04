#=
  TODO:

  Arreglar que cuando todas las layers son de la misma dimension ( {N,M} )
  el vector layers es de tipo AbstractVector{Layer{N,M}}
  en lugar de AbstractVector{Layer}. Redefinir Layer para no usar tipos paramétricos?
=#

"""
    MLP( (n₁, ), [(n₂, [fn₂])], ... , (nₘ, [fnₘ]); [T::Type] )
    MLP( n₁, [fn₁], [n₂], [fn₂], ... , nₘ, [fnₘ]; [T::Type] )
    MLP( layers::AbstractVector{Layer} )

Constructs a multilayer perceptron (MLP).
  nᵢ:  Number of neurons in the `i`th layer.
  fnᵢ: Activation function for the `i`th layer.
  T:   Type of the weight matrices. Must be such that `ones(T, (M,N))` is defined.

An `MLP` is callable, just like a `Layer` (see `Layer`'s docstring).
Additionally, you can get the activation of the `n`th layer by passing the `layernumber` keyword as follows.

    mlp(inputvector, layernumber=n)
"""
struct MLP
    layers::AbstractVector{Layer}

    function MLP(layers::AbstractVector{Layer})
      for i=2:length(layers)
        if size(layers[i])[1] != size(layers[i-1])[2]
          throw(DimensionMismatch("Layer $(i-1) and layer $i are incompatible."))
        end
      end

      return new(layers)
    end
end

function MLP(
  layerdefinitions::Vararg{Tuple};
  T::Type=Float64)

    layers = Vector{Layer}()
    n = length(layerdefinitions)

    if n<2
      throw(ArgumentError("layerdefinitions should at least contain two layers (input and output)"))
    end

    if length(layerdefinitions[1]) > 1
      info("No activation function is computed for the input layer. It's useless to specify it.")
    end

    for i=2:n
      push!(
        layers,
        Layer{layerdefinitions[i-1][1], layerdefinitions[i][1]}(
          layerdefinitions[i][2:end]...,
          T=T))
    end

    return MLP(layers)
end

function MLP(
  definitions...;
  T::Type=Float64)

    layerdefinitions = []
    i = length(definitions)

    while i>0

      if isa(definitions[i], Integer) # if is a number of neurons
        unshift!(layerdefinitions, (definitions[i], ))
        i = i - 1
      elseif method_exists(definitions[i], (Float64,)) # if is an activation function
        if !isa(definitions[i-1], Integer)
          throw(ArgumentError("Expected a list of Integers and optional activation functions"))
        else
          unshift!(layerdefinitions, (definitions[i-1], definitions[i]))
          i = i - 2
        end
      else
        throw(ArgumentError("Expected a list of Integers and optional activation functions"))
      end

    end

    return MLP(layerdefinitions...)
end

function (mlp::MLP)(
  input::AbstractVector;
  layernumber::Integer=length(mlp.layers))

    if layernumber == 1
      return mlp.layers[1](input)
    else
      return mlp.layers[layernumber](mlp(input, layernumber=layernumber-1))
    end
end

function (mlp::MLP)(
  input::Vararg{Real};
  layernumber::Integer=length(mlp.layers))

    return mlp([input...], layernumber=layernumber)
end
