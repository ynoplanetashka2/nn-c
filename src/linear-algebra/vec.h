#ifndef VEC_H
#define VEC_H

typedef struct {
  unsigned int size;
  float* values;
} vec;

vec vec_create(unsigned int size); 

void vec_free(vec vector);

vec vec_sum(const vec vec1, const vec vec2);

void vec_sum_inplace(vec* vec1, const vec vec2);

vec vec_scalar_multiply(const vec vector, float scalar);

void vec_scalar_multiply_inplace(vec* vector, float scalar);

float vec_dot_product(const vec vec1, const vec vec2);

float vec_sum_entries(const vec vector);

vec vec_zeros(unsigned int size);

vec vec_rand(unsigned int size);

vec vec_copy(const vec vector);

void vec_assign(vec* target, vec source);

vec vec_subtract(const vec vec1, const vec vec2);

void vec_subtract_inplace(vec* vec1, const vec vec2);

#endif