#include "sort.h"

void printDebug(uint32_t *arr, int n, const char *debugMessage) {
  printf("%s\n", debugMessage);
  for (int i = 0; i < n; i += 1) {
    printf("%d ", arr[i]);
  }
  printf("\n");
}

void compareThrust(uint32_t *out, int n, uint32_t *thrustOut) {
  for (int i = 0; i < n; i++) {
    if (out[i] != thrustOut[i]) {
      printf("INCORRECT :(\n");
      return;
    }
  }
  printf("CORRECT :)\n");
}