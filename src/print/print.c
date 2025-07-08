#include <stdio.h>
#include "print.h"

void print_vec(const vec vector) {
  printf("vec [ ");
  for (unsigned int i = 0; i < vector.size; ++i) {
    printf("%f", vector.values[i]);
    if (i + 1 < vector.size) {
      printf(" ");
    }
  }
  printf(" ]");
}

void print_matrix(const matrix mat) {
  printf("matrix [\n");
  for (unsigned int i = 0; i < mat.height; ++i) {
    printf("[ ");
    for (unsigned int j = 0; j < mat.width; ++j) {
      printf("%f", mat.values[i][j]);
      if (j + 1 < mat.width) {
        printf(" ");
      }
    }
    printf(" ]");
    printf("\n");
  }
  printf("]");
}