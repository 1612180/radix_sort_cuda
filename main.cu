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

void printDeviceInfo() {
  cudaDeviceProp devProv;
  CHECK(cudaGetDeviceProperties(&devProv, 0));
  printf("**********GPU info**********\n");
  printf("Name: %s\n", devProv.name);
  printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
  printf("Num SMs: %d\n", devProv.multiProcessorCount);
  printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor);
  printf("Max num warps per SM: %d\n",
         devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
  printf("GMEM: %zu byte\n", devProv.totalGlobalMem);
  printf("SMEM per SM: %zu byte\n", devProv.sharedMemPerMultiprocessor);
  printf("SMEM per block: %zu byte\n", devProv.sharedMemPerBlock);
  printf("****************************\n");
}

void timeSortThrust(uint32_t *in, int n, uint32_t *out) {
  GpuTimer timer;
  timer.Start();

  printf("\nRadix sort thrust\n");
  sortThrust(in, n, out);

  timer.Stop();
  printf("Time: %.3f ms\n", timer.Elapsed());
}

void timeSortBase1(uint32_t *in, int n, uint32_t *out, int nBits) {
  GpuTimer timer;
  timer.Start();

  printf("\nRadix sort base 1\n");
  sortBase1(in, n, out, nBits);

  timer.Stop();
  printf("Time: %.3f ms\n", timer.Elapsed());
}

void timeSortBase2(uint32_t *in, int n, uint32_t *out, int nBits,
                   int *blockSizes) {
  GpuTimer timer;
  timer.Start();

  printf("\nRadix sort base 2\n");
  sortBase2(in, n, out, nBits, blockSizes);

  timer.Stop();
  printf("Time: %.3f ms\n", timer.Elapsed());
}

void timeSortBase3(uint32_t *in, int n, uint32_t *out, int nBits,
                   int *blockSizes) {
  GpuTimer timer;
  timer.Start();

  printf("\nRadix sort base 3\n");
  sortBase3(in, n, out, nBits, blockSizes);

  timer.Stop();
  printf("Time: %.3f ms\n", timer.Elapsed());
}

void timeSortBase41(uint32_t *in, int n, uint32_t *out, int nBits,
                    int *blockSizes) {
  GpuTimer timer;
  timer.Start();

  printf("\nRadix sort base 41\n");
  sortBase41(in, n, out, nBits, blockSizes);

  timer.Stop();
  printf("Time: %.3f ms\n", timer.Elapsed());
}

int main(int argc, char **argv) {
  printDeviceInfo();

  int n = (1 << 24) + 1;
  // n = 10;
  printf("\nInput size: %d\n", n);

  size_t bytes = n * sizeof(uint32_t);
  uint32_t *in = (uint32_t *)malloc(bytes);
  uint32_t *out = (uint32_t *)malloc(bytes);

  // Use thrust for compare
  uint32_t *thrustOut = (uint32_t *)malloc(bytes);

  for (int i = 0; i < n; i++) {
    in[i] = rand();
  }

  int nBits = 4; // Default
  if (argc > 1) {
    nBits = atoi(argv[1]);
  }
  printf("\nNum bits per digit: %d\n", nBits);

  int blockSizes[3] = {512, 512, 512};
  if (argc == 5) {
    blockSizes[0] = atoi(argv[2]);
    blockSizes[1] = atoi(argv[3]);
    blockSizes[2] = atoi(argv[4]);
  }
  for (int i = 0; i < 3; i += 1) {
    printf("\nBlocksize %d: %d\n", i, blockSizes[i]);
  }

  timeSortThrust(in, n, thrustOut);

  timeSortBase1(in, n, out, nBits);
  compareThrust(out, n, thrustOut);

  timeSortBase2(in, n, out, nBits, blockSizes);
  compareThrust(out, n, thrustOut);

  timeSortBase3(in, n, out, nBits, blockSizes);
  compareThrust(out, n, thrustOut);

  timeSortBase41(in, n, out, nBits, blockSizes);
  compareThrust(out, n, thrustOut);

  free(in);
  free(out);
  free(thrustOut);

  return EXIT_SUCCESS;
}