#include <stdlib.h>

float rand_float_positive() {
  return (float) rand() / RAND_MAX;
}

float rand_float() {
  return 2 * rand_float_positive() - 1;
}