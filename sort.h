#ifndef SORT_H
#define SORT_H

#include <stdint.h>
#include <stdio.h>

#define CHECK(call)                                                            \
  {                                                                            \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                   \
      fprintf(stderr, "code: %d, reason: %s\n", error,                         \
              cudaGetErrorString(error));                                      \
      exit(1);                                                                 \
    }                                                                          \
  }

void printDebug(uint32_t *arr, int n, const char *debugMessage);
void compareThrust(uint32_t *out, int n, uint32_t *thrustOut);

void sortThrust(uint32_t *in, int n, uint32_t *out);
void sortBase1(uint32_t *in, int n, uint32_t *out, int nBits);
void sortBase2(uint32_t *in, int n, uint32_t *out, int nBits, int *blockSizes);
void sortBase3(uint32_t *in, int n, uint32_t *out, int nBits, int *blockSizes);
void sortBase41(uint32_t *in, int n, uint32_t *out, int nBits, int *blockSizes);
void sortBase42(uint32_t *in, int n, uint32_t *out, int nBits, int *blockSizes);

#endif