#include "matrix.h"
#include <stdlib.h>
#include "../rand_float.h"

matrix matrix_create(unsigned int height, unsigned int width) {
  float** rows = (float**) malloc(sizeof(float*) * height);
  for (unsigned int i = 0; i < height; ++i) {
    rows[i] = (float*) malloc(sizeof(float) * width);
  }
  return (matrix) {
    .height = height,
    .width = width,
    .values = rows
  };
}

void matrix_free(matrix mat) {
  free(mat.values);
}

vec matrix_apply(const matrix mat, const vec arg) {
  vec result = vec_create(mat.height);
  for (unsigned int i = 0; i < mat.height; ++i) {
    float sum = 0;
    for (unsigned int j = 0; j < mat.width; ++j) {
      sum += mat.values[i][j] * arg.values[j];
    }
    result.values[i] = sum;
  }
  return result;
}

void matrix_apply_inplace(const matrix mat, vec* arg) {
  const vec result = matrix_apply(mat, *arg);
  vec_free(*arg);
  arg->size = result.size;
  arg->values = result.values;
}

matrix matrix_multiply(const matrix mat1, const matrix mat2) {
  matrix result = matrix_create(mat1.height, mat2.width);
  for (unsigned int i = 0; i < mat1.height; ++i) {
    for (unsigned int j = 0; j < mat2. width; ++j) {
      float sum = 0;
      for (unsigned int k = 0; k < mat1.width; ++k) {
        sum += mat1.values[i][k] * mat2.values[k][j];
      }
      result.values[i][j] = sum;
    }
  }
  return result;
}

void matrix_multiply_inplace(matrix* mat1, const matrix mat2) {
  const matrix result = matrix_multiply(*mat1, mat2);
  matrix_free(*mat1);
  mat1->height = result.height;
  mat1->width = result.width;
  mat1->values = result.values;
}

matrix matrix_zeros(unsigned int height, unsigned int width) {
  matrix result = matrix_create(height, width);
  for (unsigned int i = 0; i < height; ++i) {
    for (unsigned int j = 0; j < width; ++j) {
      result.values[i][j] = 0;
    }
  }
  return result;
}

matrix matrix_rand(unsigned int height, unsigned int width) {
  matrix result = matrix_create(height, width);
  for (unsigned int i = 0; i < height; ++i) {
    for (unsigned int j = 0; j < width; ++j) {
      result.values[i][j] = rand_float();
    }
  }
  return result;
}

matrix matrix_transpose(const matrix mat) {
  matrix result = matrix_create(mat.width, mat.height);
  for (unsigned int i = 0; i < mat.height; ++i) {
    for (unsigned int j = 0; j < mat.width; ++j) {
      result.values[j][i] = mat.values[i][j];
    }
  }
  return result;
}

void matrix_transpose_inplace(matrix* mat) {
  const matrix result = matrix_transpose(*mat);
  matrix_free(*mat);
  mat->height = result.height;
  mat->width = result.width;
  mat->values = result.values;
}

matrix matrix_identity(unsigned int size) {
  matrix result = matrix_create(size, size);
  for (unsigned int i = 0; i < size; ++i) {
    for (unsigned int j = 0; j < size; ++j) {
      if (i == j) {
        result.values[i][j] = 1;
      } else {
        result.values[i][j] = 0;
      }
    }
  }
  return result;
}

matrix matrix_add(const matrix mat1, const matrix mat2) {
  matrix result = matrix_create(mat1.height, mat1.width);
  for (unsigned int i = 0; i < mat1.height; ++i) {
    for (unsigned int j = 0; j < mat1.width; ++j) {
      result.values[i][j] = mat1.values[i][j] + mat2.values[i][j];
    }
  }
  return result;
}

void matrix_add_inplace(matrix* mat1, const matrix mat2) {
  for (unsigned int i = 0; i < mat1->height; ++i) {
    for (unsigned int j = 0; j < mat1->width; ++j) {
      mat1->values[i][j] += mat2.values[i][j];
    }
  }
}

matrix matrix_subtract(const matrix mat1, const matrix mat2) {
  matrix result = matrix_create(mat1.height, mat1.width);
  for (unsigned int i = 0; i < mat1.height; ++i) {
    for (unsigned int j = 0; j < mat1.width; ++j) {
      result.values[i][j] = mat1.values[i][j] - mat2.values[i][j];
    }
  }
  return result;
}

void matrix_subtract_inplace(matrix* mat1, const matrix mat2) {
  for (unsigned int i = 0; i < mat1->height; ++i) {
    for (unsigned int j = 0; j < mat1->width; ++j) {
      mat1->values[i][j] -= mat2.values[i][j];
    }
  }
}

matrix matrix_scalar_multiply(const matrix mat, const float scalar) {
  matrix result = matrix_create(mat.height, mat.width);
  for (unsigned int i = 0; i < mat.height; ++i) {
    for (unsigned int j = 0; j < mat.width; ++j) {
      result.values[i][j] = mat.values[i][j] * scalar;
    }
  }
  return result;
}

void matrix_scalar_multiply_inplace(matrix* mat, const float scalar) {
  for (unsigned int i = 0; i < mat->height; ++i) {
    for (unsigned int j = 0; j < mat->width; ++j) {
      mat->values[i][j] *= scalar;
    }
  }
}