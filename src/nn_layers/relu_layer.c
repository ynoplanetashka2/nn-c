#include "relu_layer.h"

float RELU(float arg) {
  if (arg < 0) {
    return 0;
  }
  return arg;
}

float RELU_derivative(float arg) {
  if (arg < 0) {
    return 0;
  }
  return 1;
}

vec transform(unsigned int input_size, unsigned int output_size, vec vector) {
  vec result = vec_create(output_size);
  for (unsigned int i = 0; i < input_size; ++i) {
    result.values[i] = RELU(vector.values[i]);
  }
  return result;
}

matrix transform_derivative(unsigned int input_size, unsigned int output_size, vec vector) {
  matrix result = matrix_zeros(output_size, output_size);
  for (unsigned int i = 0; i < input_size; ++i) {
    result.values[i][i] = RELU_derivative(vector.values[i]);
  }
  return result;
}

nn_layer createReluLayer(unsigned int size) {
  return (nn_layer) {
    .input_size = size,
    .output_size = size,
    .transform = transform,
    .transform_derivative = transform_derivative
  };
}
