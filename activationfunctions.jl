"""
  `sigmoid(x)`

Sigmoid activation function typically used in ANNs.
"""
function sigmoid(x)
    return 1/(1 + exp(-x))
end
