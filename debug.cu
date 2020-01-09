#include "sort.h"

void printDebug(uint32_t *arr, int n, const char *debugMessage) {
  printf("%s\n", debugMessage);
  for (int i = 0; i < n; i += 1) {
    printf("%d ", arr[i]);
  }
  printf("\n");
}

void compareArray(uint32_t *arr1, int n, uint32_t *arr2) {
  for (int i = 0; i < n; i++) {
    if (arr1[i] != arr2[i]) {
      printf("INCORRECT :(\n");
      return;
    }
  }
  printf("CORRECT :)\n");
}