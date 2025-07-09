#ifndef MATRIX_H
#define MATRIX_H

#include "vec.h"
#include "real.h"

typedef struct {
  unsigned int width;
  unsigned int height;
  real** values;
} matrix;

matrix matrix_create(unsigned int height, unsigned int width);

void matrix_free(matrix mat);

vec matrix_apply(const matrix mat, const vec arg);

void matrix_apply_inplace(const matrix mat, vec* arg);

matrix matrix_multiply(const matrix mat1, const matrix mat2);

void matrix_multiply_inplace(matrix* mat1, const matrix mat2);

matrix matrix_zeros(unsigned int height, unsigned int width);

matrix matrix_rand(unsigned int height, unsigned int width);

matrix matrix_transpose(matrix mat);

void matrix_transpose_inplace(matrix* mat);

matrix matrix_identity(unsigned int size);

matrix matrix_add(const matrix mat1, const matrix mat2);

void matrix_add_inplace(matrix* mat1, const matrix mat2);

matrix matrix_subtract(const matrix mat1, const matrix mat2);

void matrix_subtract_inplace(matrix* mat1, const matrix mat2);

matrix matrix_scalar_multiply(const matrix mat, const real scalar);

void matrix_scalar_multiply_inplace(matrix* mat, const real scalar);

#endif