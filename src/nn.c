#include <malloc.h>
#include <stdio.h>
#include <math.h>
#include "nn.h"
#include "square_error/square_error.h"

nn nn_create(nn_layer* layers, unsigned int layers_count, unsigned int input_size, unsigned int output_size) {
  matrix* weights = (matrix*) malloc(sizeof(matrix) * (layers_count + 1));
  unsigned int weights_input_size = input_size;
  unsigned int weights_output_size = layers_count > 0 ? layers[0].input_size : output_size;
  for (unsigned int i = 0; i < layers_count; ++i) {
    weights[i] = matrix_create(weights_input_size, weights_output_size);
    weights_input_size = layers[i].output_size;
    weights_output_size = i + 1 < layers_count ? layers[i + 1].input_size : output_size;
  }
  weights[layers_count] = matrix_create(weights_input_size, weights_output_size);

  vec* bias = (vec*) malloc(sizeof(vec) * (layers_count + 1));
  for (unsigned int i = 0; i < layers_count; ++i) {
    const nn_layer layer = layers[i];
    bias[i] = vec_create(layer.input_size);
  }
  bias[layers_count] = vec_create(output_size);

  return (nn) {
    .layers_count = layers_count,
    .input_size = input_size,
    .output_size = output_size,
    .layers = layers,
    .weights = weights,
    .bias = bias
  };
}

void nn_free(nn nn_instance) {
  const unsigned int layers_count = nn_instance.layers_count;
  for (unsigned int i = 0; i < layers_count + 1; ++i) {
    vec_free(nn_instance.bias[i]);
    matrix_free(nn_instance.weights[i]);
  }
  free(nn_instance.bias);
  free(nn_instance.weights);
  free(nn_instance.layers);
}

nn nn_zeros(nn_layer* layers, unsigned int layers_count, unsigned int input_size, unsigned int output_size) {
  matrix* weights = (matrix*) malloc(sizeof(matrix) * (layers_count + 1));
  unsigned int weights_input_size = input_size;
  unsigned int weights_output_size = layers_count > 0 ? layers[0].input_size : output_size;
  for (unsigned int i = 0; i < layers_count; ++i) {
    weights[i] = matrix_zeros(weights_input_size, weights_output_size);
    weights_input_size = layers[i].output_size;
    weights_output_size = i + 1 < layers_count ? layers[i + 1].input_size : output_size;
  }
  weights[layers_count] = matrix_zeros(weights_input_size, weights_output_size);

  vec* bias = (vec*) malloc(sizeof(vec) * (layers_count + 1));
  for (unsigned int i = 0; i < layers_count; ++i) {
    const nn_layer layer = layers[i];
    bias[i] = vec_zeros(layer.input_size);
  }
  bias[layers_count] = vec_zeros(output_size);

  return (nn) {
    .layers_count = layers_count,
    .input_size = input_size,
    .output_size = output_size,
    .layers = layers,
    .weights = weights,
    .bias = bias
  };
}

nn nn_rand(nn_layer* layers, unsigned int layers_count, unsigned int input_size, unsigned int output_size) {
  matrix* weights = (matrix*) malloc(sizeof(matrix) * (layers_count + 1));
  unsigned int weights_input_size = input_size;
  unsigned int weights_output_size = layers_count > 0 ? layers[0].input_size : output_size;
  for (unsigned int i = 0; i < layers_count; ++i) {
    weights[i] = matrix_rand(weights_input_size, weights_output_size);
    weights_input_size = layers[i].output_size;
    weights_output_size = i + 1 < layers_count ? layers[i + 1].input_size : output_size;
  }
  weights[layers_count] = matrix_rand(weights_input_size, weights_output_size);

  vec* bias = (vec*) malloc(sizeof(vec) * (layers_count + 1));
  for (unsigned int i = 0; i < layers_count; ++i) {
    const nn_layer layer = layers[i];
    bias[i] = vec_rand(layer.input_size);
  }
  bias[layers_count] = vec_rand(output_size);

  return (nn) {
    .layers_count = layers_count,
    .input_size = input_size,
    .output_size = output_size,
    .layers = layers,
    .weights = weights,
    .bias = bias
  };
}

matrix _compute_weights_gradient(
  const matrix weights,
  const vec signal,
  const vec expected_output,
  const vec bias,
  const vec activation_function_value,
  const matrix activation_function_derivative_value
) {
  matrix transposed_activation_function_derivative = matrix_transpose(activation_function_derivative_value);
  vec rhs = vec_subtract(activation_function_value, expected_output);
  matrix_apply_inplace(transposed_activation_function_derivative, &rhs);
  matrix gradient = matrix_create(weights.height, weights.width);
  for (unsigned int i = 0; i < weights.height; ++i) {
    for (unsigned int j = 0; j < weights.width; ++j) {
      gradient.values[i][j] = signal.values[j] * rhs.values[i];
    }
  }

  matrix_free(transposed_activation_function_derivative);
  vec_free(rhs);

  return gradient;
}

vec _compute_signal_gradient(
  const matrix weights,
  const vec signal,
  const vec expected_output,
  const vec bias,
  const vec activation_function_value,
  const matrix activation_function_derivative_value
) {
  matrix transposed_activation_function_derivative_value = matrix_transpose(activation_function_derivative_value);
  matrix transposed_weights = matrix_transpose(weights);
  vec rhs = vec_subtract(activation_function_value, expected_output);
  matrix_apply_inplace(transposed_activation_function_derivative_value, &rhs);
  matrix_apply_inplace(transposed_weights, &rhs);

  matrix_free(transposed_activation_function_derivative_value);
  matrix_free(transposed_weights);

  return rhs;
}

vec _compute_bias_gradient(
  const matrix weights,
  const vec signal,
  const vec expected_output,
  const vec bias,
  const vec activation_function_value,
  const matrix activation_function_derivative_value
) {
  matrix transposed_activation_function_derivative = matrix_transpose(activation_function_derivative_value);
  vec rhs = vec_subtract(activation_function_value, expected_output);
  matrix_apply_inplace(transposed_activation_function_derivative, &rhs);

  matrix_free(transposed_activation_function_derivative);

  return rhs;
}

typedef struct {
  /**
   * Output signals for corresponding layers (including output layer)
   */
  vec* nn_prediction_results;
  /**
   * Derivative values of the activation function for corresponding layers (excluding output layer, since no activation function)
   */
  matrix* nn_transform_derivative_values;
} _nn_prediction_result_for_training;

void _nn_free_prediction_result_for_training(_nn_prediction_result_for_training* prediction_results, const unsigned int layers_count) {
  for (unsigned int i = 0; i < layers_count + 1; ++i) {
    vec_free(prediction_results->nn_prediction_results[i]);
    if (i < layers_count) {
      matrix_free(prediction_results->nn_transform_derivative_values[i]);
    }
  }
  free(prediction_results->nn_prediction_results);
}

_nn_prediction_result_for_training _nn_compute_prediction_result_for_training(
  const nn nn_instance,
  const vec input
) {
  const unsigned int layers_count = nn_instance.layers_count;
  vec buffer = vec_copy(input);
  vec* nn_prediction_results = (vec*) malloc(sizeof(vec) * (layers_count + 1));
  matrix* nn_transform_derivative_values = (matrix*) malloc(sizeof(matrix) * layers_count);
  for (unsigned int i = 0; i < nn_instance.layers_count; ++i) {
    matrix weights = nn_instance.weights[i];
    vec bias = nn_instance.bias[i];
    nn_layer layer = nn_instance.layers[i];
    matrix_apply_inplace(weights, &buffer);
    vec_sum_inplace(&buffer, bias);
    matrix transform_derivative_value = layer.transform_derivative(layer.input_size, layer.output_size, buffer);
    nn_transform_derivative_values[i] = transform_derivative_value;
    vec_assign(&buffer, layer.transform(layer.input_size, layer.output_size, buffer));
    nn_prediction_results[i] = vec_copy(buffer);
  }
  matrix weights = nn_instance.weights[layers_count];
  vec bias = nn_instance.bias[layers_count];
  matrix_apply_inplace(weights, &buffer);
  vec_sum_inplace(&buffer, bias);
  nn_prediction_results[layers_count] = buffer;
  _nn_prediction_result_for_training result = {
    .nn_prediction_results = nn_prediction_results,
    .nn_transform_derivative_values = nn_transform_derivative_values
  };
  return result;
}

void _nn_fit_once(nn nn_instance, const vec input, const vec expected_output, const float epsilon) {
  const unsigned int layers_count = nn_instance.layers_count;
  _nn_prediction_result_for_training nn_prediction_results = _nn_compute_prediction_result_for_training(nn_instance, input);
  vec current_expected_output = vec_copy(expected_output);
  matrix identity = matrix_identity(nn_instance.output_size);

  if (layers_count > 0) {
    {
      vec bias_gradient = _compute_bias_gradient(
        nn_instance.weights[nn_instance.layers_count],
        nn_prediction_results.nn_prediction_results[nn_instance.layers_count - 1],
        current_expected_output,
        nn_instance.bias[nn_instance.layers_count],
        nn_prediction_results.nn_prediction_results[nn_instance.layers_count],
        identity
      );
      vec signal_gradient = _compute_signal_gradient(
        nn_instance.weights[nn_instance.layers_count],
        nn_prediction_results.nn_prediction_results[nn_instance.layers_count - 1],
        current_expected_output,
        nn_instance.bias[nn_instance.layers_count],
        nn_prediction_results.nn_prediction_results[nn_instance.layers_count],
        identity
      );
      matrix weights_gradient = _compute_weights_gradient(
        nn_instance.weights[nn_instance.layers_count],
        nn_prediction_results.nn_prediction_results[nn_instance.layers_count - 1],
        current_expected_output,
        nn_instance.bias[nn_instance.layers_count],
        nn_prediction_results.nn_prediction_results[nn_instance.layers_count],
        identity
      );
      vec_scalar_multiply_inplace(&bias_gradient, epsilon);
      matrix_scalar_multiply_inplace(&weights_gradient, epsilon);
      vec_subtract_inplace(&nn_instance.bias[layers_count], bias_gradient);
      matrix_subtract_inplace(&nn_instance.weights[layers_count], weights_gradient);

      vec_assign(&current_expected_output, 
        vec_subtract(nn_prediction_results.nn_prediction_results[layers_count - 1], signal_gradient));
      vec_free(signal_gradient);
      vec_free(bias_gradient);
      matrix_free(weights_gradient);
    }
    for (unsigned int i = nn_instance.layers_count - 1; i >= 1; --i) {
      const vec activation_function_value = nn_prediction_results.nn_prediction_results[i];
      const matrix activation_function_derivative_value = nn_prediction_results.nn_transform_derivative_values[i];
      vec bias_gradient = _compute_bias_gradient(
        nn_instance.weights[i],
        nn_prediction_results.nn_prediction_results[i - 1],
        current_expected_output,
        nn_instance.bias[i],
        activation_function_value,
        activation_function_derivative_value
      );
      vec signal_gradient = _compute_signal_gradient(
        nn_instance.weights[i],
        nn_prediction_results.nn_prediction_results[i - 1],
        current_expected_output,
        nn_instance.bias[i],
        activation_function_value,
        activation_function_derivative_value
      );
      matrix weights_gradient = _compute_weights_gradient(
        nn_instance.weights[i],
        nn_prediction_results.nn_prediction_results[i - 1],
        current_expected_output,
        nn_instance.bias[i],
        activation_function_value,
        activation_function_derivative_value
      );
      vec_scalar_multiply_inplace(&bias_gradient, epsilon);
      matrix_scalar_multiply_inplace(&weights_gradient, epsilon);
      vec_subtract_inplace(&nn_instance.bias[layers_count], bias_gradient);
      matrix_subtract_inplace(&nn_instance.weights[layers_count], weights_gradient);

      vec_assign(&current_expected_output,
        vec_subtract(nn_prediction_results.nn_prediction_results[i - 1], signal_gradient));
      vec_free(signal_gradient);
      vec_free(bias_gradient);
      matrix_free(weights_gradient);
    }
    {
      const vec activation_function_value = nn_prediction_results.nn_prediction_results[0];
      const matrix activation_function_derivative_value = nn_prediction_results.nn_transform_derivative_values[0];
      vec bias_gradient = _compute_bias_gradient(
        nn_instance.weights[0],
        input,
        current_expected_output,
        nn_instance.bias[0],
        activation_function_value,
        activation_function_derivative_value
      );
      matrix weights_gradient = _compute_weights_gradient(
        nn_instance.weights[nn_instance.layers_count],
        nn_prediction_results.nn_prediction_results[nn_instance.layers_count - 1],
        current_expected_output,
        nn_instance.bias[nn_instance.layers_count],
        activation_function_value,
        activation_function_derivative_value
      );
      vec_scalar_multiply_inplace(&bias_gradient, epsilon);
      matrix_scalar_multiply_inplace(&weights_gradient, epsilon);
      vec_subtract_inplace(&nn_instance.bias[layers_count], bias_gradient);
      matrix_subtract_inplace(&nn_instance.weights[layers_count], weights_gradient);

      vec_free(bias_gradient);
      matrix_free(weights_gradient);
    }
  } else {
    const matrix weights = nn_instance.weights[0];
    const vec bias = nn_instance.bias[0];
    const vec activation_function_value = nn_prediction_results.nn_prediction_results[0];

    vec bias_gradient = _compute_bias_gradient(
      weights,
      input,
      current_expected_output,
      bias,
      activation_function_value,
      identity
    );

    matrix weights_gradient = _compute_weights_gradient(
      weights,
      input,
      current_expected_output,
      bias,
      activation_function_value,
      identity
    );
    vec_scalar_multiply_inplace(&bias_gradient, epsilon);
    matrix_scalar_multiply_inplace(&weights_gradient, epsilon);
    vec_subtract_inplace(&nn_instance.bias[0], bias_gradient);
    matrix_subtract_inplace(&nn_instance.weights[0], weights_gradient);

    vec_free(bias_gradient);
    matrix_free(weights_gradient);
  }
  
  vec_free(current_expected_output);
  _nn_free_prediction_result_for_training(&nn_prediction_results, layers_count);
  matrix_free(identity);
}

void nn_fit(nn nn_instance, const vec input, const vec expected_output, unsigned int training_cycles) {
  for (unsigned int cycle = 0; cycle < training_cycles; ++cycle) {
    const float epsilon = 1.0f / log(cycle + 2) / 1000.0f;
    _nn_fit_once(nn_instance, input, expected_output, epsilon);
    #ifdef DEBUG
    if (cycle % 100 == 0) {
      vec nn_prediction_results = nn_predict(nn_instance, input);
      float error = square_error_vec(nn_prediction_results, expected_output);
      printf("Cycle %u: Error = %f\n", cycle, error);
      vec_free(nn_prediction_results);
    }
    #endif
  }
}

vec nn_predict(const nn nn_instance, const vec input) {
  const unsigned int layers_count = nn_instance.layers_count;
  vec buffer = vec_copy(input);
  for (unsigned int i = 0; i < nn_instance.layers_count; ++i) {
    matrix weights = nn_instance.weights[i];
    vec bias = nn_instance.bias[i];
    nn_layer layer = nn_instance.layers[i];
    matrix_apply_inplace(weights, &buffer);
    vec_sum_inplace(&buffer, bias);
    vec_assign(&buffer, layer.transform(layer.input_size, layer.output_size, buffer));
  }
  matrix weights = nn_instance.weights[layers_count];
  vec bias = nn_instance.bias[layers_count];
  matrix_apply_inplace(weights, &buffer);
  vec_sum_inplace(&buffer, bias);
  return buffer;
}
