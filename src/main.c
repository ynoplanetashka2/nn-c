#include <stdio.h>
#include "nn.h"
#include "nn_layers/relu_layer.h"
#include "print/print.h"

#include <float.h>

int main(void) {
  #ifdef DEBUG
  // Unmask exceptions for invalid, divide-by-zero, and overflow
  _controlfp(_controlfp(0, 0) & ~(_EM_ZERODIVIDE | _EM_INVALID | _EM_OVERFLOW), _MCW_EM);  // Enable exceptions for divide-by-zero, invalid operations, and overflow
  #endif

  nn_layer layers[] = {
    createReluLayer(10),
    createReluLayer(10)
  };
  nn nn_instance = nn_rand(layers, 2, 10, 10);
  vec input = vec_create(10);
  vec output = vec_create(10);
  for (unsigned int i = 0; i < 10; ++i) {
    const float x = (float)i;
    input.values[i] = i;
    output.values[i] = x * x;
  }
  nn_fit(nn_instance, input, output, 1000);
  vec_free(input);
  vec_free(output);
  vec test_input = vec_create(10);
  for (unsigned int i = 0; i < 10; ++i) {
    test_input.values[i] = i - 10;
  }
  vec result = nn_predict(nn_instance, test_input);
  print_vec(result);
  printf("\n");
  
  vec_free(test_input);
  vec_free(result);
  nn_free(nn_instance);

  return 0;
}