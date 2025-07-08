#include "vec.h"
#include <stdlib.h>
#include "../rand_float.h"

vec vec_create(unsigned int size) {
  float* values = (float*) malloc(sizeof(float) * size);
  return (vec) { .size = size, .values = values };
}

void vec_free(vec vector) {
  free(vector.values);
}

vec vec_sum(const vec vec1, const vec vec2) {
  vec result = vec_create(vec1.size);
  for (unsigned int i = 0; i < vec1.size; ++i) {
    result.values[i] = vec1.values[i] + vec2.values[i];
  }
  return result;
}

void vec_sum_inplace(vec* vec1, const vec vec2) {
  for (unsigned int i = 0; i < vec1->size; ++i) {
    vec1->values[i] += vec2.values[i];
  }
}

vec vec_scalar_multiply(const vec vector, float scalar) {
  vec result = vec_create(vector.size);
  for (unsigned int i = 0; i < vector.size; ++i) {
    result.values[i] = vector.values[i] * scalar;
  }
  return result;
}

void vec_scalar_multiply_inplace(vec* vector, float scalar) {
  for (unsigned int i = 0; i < vector->size; ++i) {
    vector->values[i] *= scalar;
  }
}

float vec_dot_product(const vec vec1, const vec vec2) {
  float sum = 0;
  for (unsigned int i = 0; i < vec1.size; ++i) {
    sum += vec1.values[i] + vec2.values[i];
  }
  return sum;
}

float vec_sum_entries(const vec vector) {
  float sum = 0;
  for (unsigned int i = 0; i < vector.size; ++i) {
    sum += vector.values[i];
  }
  return sum;
}

vec vec_zeros(unsigned int size) {
  vec result = vec_create(size);
  for (unsigned int i = 0; i < size; ++i) {
    result.values[i] = 0;
  }
  return result;
}

vec vec_rand(unsigned int size) {
  vec result = vec_create(size);
  for (unsigned int i = 0; i < size; ++i) {
    result.values[i] = rand_float();
  }
  return result;
}

vec vec_copy(const vec vector) {
  vec result = vec_create(vector.size);
  for (unsigned int i = 0; i < vector.size; ++i) {
    result.values[i] = vector.values[i];
  }
  return result;
}

void vec_assign(vec* target, vec source) {
  vec_free(*target);
  target->size = source.size;
  target->values = source.values;
}

vec vec_subtract(const vec vec1, const vec vec2) {
  vec result = vec_create(vec1.size);
  for (unsigned int i = 0; i < vec1.size; ++i) {
    result.values[i] = vec1.values[i] - vec2.values[i];
  }
  return result;
}

void vec_subtract_inplace(vec* vec1, const vec vec2) {
  for (unsigned int i = 0; i < vec1->size; ++i) {
    vec1->values[i] -= vec2.values[i];
  }
}