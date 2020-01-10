#include "sort.h"

struct GpuTimer {
  cudaEvent_t start;
  cudaEvent_t stop;

  GpuTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void Start() {
    cudaEventRecord(start, 0);
    cudaEventSynchronize(start);
  }

  void Stop() { cudaEventRecord(stop, 0); }

  float Elapsed() {
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  }
};

void timeSortThrust(uint32_t *in, int n, uint32_t *out) {
  GpuTimer timer;
  timer.Start();

  printf("Radix sort thrust\n");
  sortThrust(in, n, out);

  timer.Stop();
  printf("Time: %.3f ms\n", timer.Elapsed());
}

void timeSortBase1(uint32_t *in, int n, uint32_t *out, int nBits) {
  GpuTimer timer;
  timer.Start();

  printf("Radix sort base 1\n");
  sortBase1(in, n, out, nBits);

  timer.Stop();
  printf("Time: %.3f ms\n", timer.Elapsed());
}

void timeSortBase2(uint32_t *in, int n, uint32_t *out, int nBits,
                   int *blockSizes) {
  GpuTimer timer;
  timer.Start();

  printf("Radix sort base 2\n");
  sortBase2(in, n, out, nBits, blockSizes);

  timer.Stop();
  printf("Time: %.3f ms\n", timer.Elapsed());
}

void timeSortBase3(uint32_t *in, int n, uint32_t *out, int nBits,
                   int *blockSizes) {
  GpuTimer timer;
  timer.Start();

  printf("Radix sort base 3\n");
  sortBase3(in, n, out, nBits, blockSizes);

  timer.Stop();
  printf("Time: %.3f ms\n", timer.Elapsed());
}

void timeSortBase4(uint32_t *in, int n, uint32_t *out, int nBits,
                    int *blockSizes) {
  GpuTimer timer;
  timer.Start();

  printf("Radix sort base 4\n");
  sortBase4(in, n, out, nBits, blockSizes);

  timer.Stop();
  printf("Time: %.3f ms\n", timer.Elapsed());
}

void timeSortBase41(uint32_t *in, int n, uint32_t *out, int nBits,
                    int *blockSizes) {
  GpuTimer timer;
  timer.Start();

  printf("Radix sort base 41\n");
  sortBase41(in, n, out, nBits, blockSizes);

  timer.Stop();
  printf("Time: %.3f ms\n", timer.Elapsed());
}

void timeSortBase42(uint32_t *in, int n, uint32_t *out, int nBits,
                    int *blockSizes) {
  GpuTimer timer;
  timer.Start();

  printf("Radix sort base 42\n");
  sortBase42(in, n, out, nBits, blockSizes);

  timer.Stop();
  printf("Time: %.3f ms\n", timer.Elapsed());
}

void timeSortBase43(uint32_t *in, int n, uint32_t *out, int nBits,
                    int *blockSizes) {
  GpuTimer timer;
  timer.Start();

  printf("Radix sort base 43\n");
  sortBase43(in, n, out, nBits, blockSizes);

  timer.Stop();
  printf("Time: %.3f ms\n", timer.Elapsed());
}

int main(int argc, char **argv) {
  int n = (1 << 24) + 1;
  // n = 10;
  printf("Input size: %d\n", n);

  size_t bytes = n * sizeof(uint32_t);
  uint32_t *in = (uint32_t *)malloc(bytes);
  uint32_t *out = (uint32_t *)malloc(bytes);

  // ensure in array not change
  uint32_t *tempIn = (uint32_t *)malloc(bytes);

  // Use thrust for compare
  uint32_t *thrustOut = (uint32_t *)malloc(bytes);

  for (int i = 0; i < n; i++) {
    in[i] = rand();
  }

  int nBits = 4; // Default
  if (argc > 1) {
    nBits = atoi(argv[1]);
  }
  printf("Num bits per digit: %d\n", nBits);

  int blockSizes[3] = {512, 512, 512};
  if (argc == 5) {
    blockSizes[0] = atoi(argv[2]);
    blockSizes[1] = atoi(argv[3]);
    blockSizes[2] = atoi(argv[4]);
  }
  for (int i = 0; i < 3; i += 1) {
    printf("Blocksize %d: %d\n", i, blockSizes[i]);
  }

  memcpy(tempIn, in, n * sizeof(uint32_t));
  timeSortThrust(tempIn, n, thrustOut);

  memcpy(tempIn, in, n * sizeof(uint32_t));
  memset(out, 0, n * sizeof(uint32_t));
  timeSortBase1(tempIn, n, out, nBits);
  compareArray(out, n, thrustOut);

  memcpy(tempIn, in, n * sizeof(uint32_t));
  memset(out, 0, n * sizeof(uint32_t));
  timeSortBase2(tempIn, n, out, nBits, blockSizes);
  compareArray(out, n, thrustOut);

  memcpy(tempIn, in, n * sizeof(uint32_t));
  memset(out, 0, n * sizeof(uint32_t));
  timeSortBase3(tempIn, n, out, nBits, blockSizes);
  compareArray(out, n, thrustOut);

  memcpy(tempIn, in, n * sizeof(uint32_t));
  memset(out, 0, n * sizeof(uint32_t));
  timeSortBase4(tempIn, n, out, nBits, blockSizes);
  compareArray(out, n, thrustOut);

  memcpy(tempIn, in, n * sizeof(uint32_t));
  memset(out, 0, n * sizeof(uint32_t));
  timeSortBase41(tempIn, n, out, nBits, blockSizes);
  compareArray(out, n, thrustOut);

  memcpy(tempIn, in, n * sizeof(uint32_t));
  memset(out, 0, n * sizeof(uint32_t));
  timeSortBase42(tempIn, n, out, nBits, blockSizes);
  compareArray(out, n, thrustOut);

  memcpy(tempIn, in, n * sizeof(uint32_t));
  memset(out, 0, n * sizeof(uint32_t));
  timeSortBase43(tempIn, n, out, nBits, blockSizes);
  compareArray(out, n, thrustOut);

  free(in);
  free(out);
  free(thrustOut);

  return EXIT_SUCCESS;
}