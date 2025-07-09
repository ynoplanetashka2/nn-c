#include <math.h>
#include "sigmoid_layer.h"
#include "../linear-algebra/vec.h"
#include "../linear-algebra/matrix.h"
#include "../linear-algebra/real.h"

real sigmoid(real x) {
  return 1.0 / (1.0 + exp(-x));
}

real sq(real x) {
  return x * x;
}

real sigmoid_derivative(real x) {
  return exp(-x) / sq(1.0 + exp(-x));
}

vec SIGMOID_transform(unsigned int input_size, unsigned int output_size, vec vector) {
  vec result = vec_create(output_size);
  for (unsigned int i = 0; i < input_size; ++i) {
    result.values[i] = sigmoid(vector.values[i]);
  }
  return result;
}

matrix SIGMOID_transform_derivative(unsigned int input_size, unsigned int output_size, vec vector) {
  matrix result = matrix_zeros(output_size, output_size);
  for (unsigned int i = 0; i < input_size; ++i) {
    result.values[i][i] = sigmoid_derivative(vector.values[i]);
  }
  return result;
}


nn_layer create_sigmoid_layer(unsigned int size) {
  return (nn_layer) {
    .input_size = size,
    .output_size = size,
    .transform = SIGMOID_transform,
    .transform_derivative = SIGMOID_transform_derivative
  };
}