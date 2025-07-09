#ifndef VEC_H
#define VEC_H

#include "real.h"

typedef struct {
  unsigned int size;
  real* values;
} vec;

vec vec_create(unsigned int size); 

void vec_free(vec vector);

vec vec_sum(const vec vec1, const vec vec2);

void vec_sum_inplace(vec* vec1, const vec vec2);

vec vec_scalar_multiply(const vec vector, real scalar);

void vec_scalar_multiply_inplace(vec* vector, real scalar);

real vec_dot_product(const vec vec1, const vec vec2);

real vec_sum_entries(const vec vector);

vec vec_zeros(unsigned int size);

vec vec_rand(unsigned int size);

vec vec_copy(const vec vector);

void vec_assign(vec* target, vec source);

vec vec_subtract(const vec vec1, const vec vec2);

void vec_subtract_inplace(vec* vec1, const vec vec2);

#endif