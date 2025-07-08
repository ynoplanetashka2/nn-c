#include "square_error.h"

float square_error_vec(const vec vec1, const vec vec2) {
  float result = 0;
  for (unsigned int i = 0; i < vec1.size; ++i) {
    result += (vec1.values[i] - vec2.values[i]) * (vec1.values[i] - vec2.values[i]);
  }
  return result;
}

vec square_error_vec_gradient(const vec actual_value, const vec expected_value) {
  vec result = vec_create(actual_value.size);
  for (unsigned int i = 0; i < result.size; ++i) {
    result.values[i] = 2 * (actual_value.values[i] - expected_value.values[i]);
  }
  return result;
}