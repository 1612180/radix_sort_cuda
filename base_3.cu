// Base 1
// Radix sort tuan tu
#include <stdint.h>
#include <stdio.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

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

// Sequential radix sort
// Assume: nBits (k in slides) in {1, 2, 4, 8, 16}
void sortByHost(const uint32_t *in, int n, uint32_t *out, int nBits) {
  int nBins = 1 << nBits; // 2^nBits
  int *hist = (int *)malloc(nBins * sizeof(int));
  int *histScan = (int *)malloc(nBins * sizeof(int));

  // In each counting sort, we sort data in "src" and write result to "dst"
  // Then, we swap these 2 pointers and go to the next counting sort
  // At first, we assign "src = in" and "dest = out"
  // However, the data pointed by "in" is read-only
  // --> we create a copy of this data and assign "src" to the address of this
  // copy
  uint32_t *src = (uint32_t *)malloc(n * sizeof(uint32_t));
  memcpy(src, in, n * sizeof(uint32_t));
  uint32_t *originalSrc = src; // Use originalSrc to free memory later
  uint32_t *dst = out;

  // Loop from LSD (Least Significant Digit) to MSD (Most Significant Digit)
  // (Each digit consists of nBits bits)
  // In each loop, sort elements according to the current digit
  // (using STABLE counting sort)
  for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits) {
    // TODO: Compute "hist" of the current digit
    memset(hist, 0, nBins * sizeof(int));
    for (int i = 0; i < n; i++) {
      int bin = (src[i] >> bit) & (nBins - 1);
      hist[bin]++;
    }

    // TODO: Scan "hist" (exclusively) and save the result to "histScan"
    histScan[0] = 0;
    for (int bin = 1; bin < nBins; bin++)
      histScan[bin] = histScan[bin - 1] + hist[bin - 1];

    // TODO: From "histScan", scatter elements in "src" to correct locations in
    // "dst"
    for (int i = 0; i < n; i++) {
      int bin = (src[i] >> bit) & (nBins - 1);
      dst[histScan[bin]] = src[i];
      histScan[bin]++;
    }

    // TODO: Swap "src" and "dst"
    uint32_t *temp = src;
    src = dst;
    dst = temp;
  }

  // TODO: Copy result to "out"
  memcpy(out, src, n * sizeof(uint32_t));
  // Free memories
  free(hist);
  free(histScan);
  free(originalSrc);
}

void printAll(uint32_t *arr, int n) {
  for (int i = 0; i < n; i += 1) {
    printf("%d ", arr[i]);
  }
  printf("\n");
}

// scan kernel on in -> out and blkSums
__global__ void scanBlkKernel(uint32_t *in, int n, uint32_t *out,
                              uint32_t *blkSums) {
  // TODO
  extern __shared__ uint32_t section[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // exclusive scan
  if (i < n && threadIdx.x != 0) {
    section[threadIdx.x] = in[i - 1];
  } else {
    section[threadIdx.x] = 0;
  }
  __syncthreads();

  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    // copy temp section[threadIdx.x - stride] before changed
    int temp = 0;
    if (stride <= threadIdx.x) {
      temp = section[threadIdx.x - stride];
    }
    __syncthreads();

    section[threadIdx.x] += temp;
    __syncthreads();
  }

  __syncthreads();
  if (i < n) {
    out[i] = section[threadIdx.x];
  }

  // copy to blkSums
  __syncthreads();
  if (blkSums != NULL && threadIdx.x == 0) {
    blkSums[blockIdx.x] = section[blockDim.x - 1];
  }

  // exclusive missing final index in 1 block "in"
  __syncthreads();
  if (i < n && threadIdx.x == blockDim.x - 1) {
    blkSums[blockIdx.x] += in[i];
  }
}

// add blkSums to out
__global__ void addScannedBlkSums(uint32_t *out, int n, uint32_t *blkSums) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  // skip first block index in "out"
  if (i < n && blockIdx.x > 0) {
    out[i] += blkSums[blockIdx.x - 1];
  }
}

__global__ void convertBinary(const uint32_t *in, int n, uint32_t *inBinary,
                              int bit) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    inBinary[i] = (in[i] >> bit) & 1;
  }
}

__global__ void countingSort(const uint32_t *in, int n, uint32_t *out,
                             const uint32_t *inBinary, const uint32_t *inScan,
                             int nZeros) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    int rank = inBinary[i] == 0 ? i - inScan[i] : nZeros + inScan[i];
    out[rank] = in[i];
  }
}

// (Partially) Parallel radix sort: implement parallel histogram and
// parallel scan in counting sort Assume: nBits (k in slides) in {1, 2, 4,
// 8, 16} Why "int * blockSizes"? Because we may want different block sizes
// for diffrent kernels:
//   blockSizes[0] for the histogram kernel
//   blockSizes[1] for the scan kernel
void sortByDevice(const uint32_t *in, int n, uint32_t *out, int nBits,
                  int *blockSizes) {
  // TODO

  // in -> inBinary
  // exclusive scan inBinary -> inScan
  // inScan[i]: count of 1 left of i index
  //
  // nZeros = n - inScan[n-1] - inBinary[n-1]
  // n is count 0 + count 1
  // inScan[n-1]: count 1 left of n - 1
  // inBinary[n-1]: 0 or 1
  //
  // calculate rank
  // inBinary[i] == 0 => rank = i - inScan[i]
  // inBinary[i] == 1 => rank = nZeros + inScan[i]
  //
  // scatter
  // out[rank] = in[i]

  uint32_t *d_in, *d_out;
  CHECK(cudaMalloc(&d_in, n * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_out, n * sizeof(uint32_t)));
  CHECK(cudaMemcpy(d_in, in, n * sizeof(uint32_t), cudaMemcpyHostToDevice));

  // scan kernel
  int scanBlockSize = blockSizes[1];
  dim3 scanGridSize((n - 1) / scanBlockSize + 1);

  uint32_t *inScan = (uint32_t *)malloc(n * sizeof(uint32_t));
  uint32_t *blkSums = (uint32_t *)malloc(scanGridSize.x * sizeof(uint32_t));

  uint32_t *d_inScan, *d_blkSums;
  CHECK(cudaMalloc(&d_inScan, n * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_blkSums, scanGridSize.x * sizeof(uint32_t)));

  // convert binary kernel
  int convertBlockSize = blockSizes[2];
  dim3 convertGridSize((n - 1) / convertBlockSize + 1);

  uint32_t *inBinary = (uint32_t *)malloc(n * sizeof(uint32_t));

  uint32_t *d_inBinary;
  CHECK(cudaMalloc(&d_inBinary, n * sizeof(uint32_t)));

  // counting sort kernel
  int countingSortBlockSize = blockSizes[3];
  dim3 countingSortGridSize((n - 1) / countingSortBlockSize + 1);

  // 1 bytes = 8 bit
  for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += 1) {
    // convert d_in to d_inBinary
    convertBinary<<<convertGridSize, convertBlockSize>>>(d_in, n, d_inBinary,
                                                         bit);
    CHECK(cudaDeviceSynchronize());

    // scan kernel on d_inBinary
    scanBlkKernel<<<scanGridSize, scanBlockSize, scanBlockSize * sizeof(int)>>>(
        d_inBinary, n, d_inScan, d_blkSums);
    CHECK(cudaDeviceSynchronize());

    // d_blkSums -> blkSums
    // inclusive scan blkSums on host
    // blkSums -> d_blkSums
    CHECK(cudaMemcpy(blkSums, d_blkSums, scanGridSize.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    for (int i = 1; i < scanGridSize.x; i += 1) {
      blkSums[i] += blkSums[i - 1];
    }
    CHECK(cudaMemcpy(d_blkSums, blkSums, scanGridSize.x * sizeof(int),
                     cudaMemcpyHostToDevice));

    // add d_blkSums to d_inBinary
    addScannedBlkSums<<<scanGridSize, scanBlockSize>>>(d_inScan, n, d_blkSums);
    CHECK(cudaDeviceSynchronize());

    // calculate nZeros
    CHECK(cudaMemcpy(inScan, d_inScan, n * sizeof(uint32_t),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(inBinary, d_inBinary, n * sizeof(uint32_t),
                     cudaMemcpyDeviceToHost));
    int nZeros = n - inScan[n - 1] - inBinary[n - 1];

    // counting sort kernel
    countingSort<<<countingSortGridSize, countingSortBlockSize>>>(
        d_in, n, d_out, d_inBinary, d_inScan, nZeros);
    CHECK(cudaDeviceSynchronize());

    uint32_t *temp = d_in;
    d_in = d_out;
    d_out = temp;
  }

  CHECK(cudaMemcpy(out, d_in, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));

  CHECK(cudaFree(d_in));
  CHECK(cudaFree(d_out));
  CHECK(cudaFree(d_inScan));
  CHECK(cudaFree(d_blkSums));
  CHECK(cudaFree(d_inBinary));

  free(inScan);
  free(blkSums);
  free(inBinary);
}

// Radix sort
void sort(const uint32_t *in, int n, uint32_t *out, int nBits,
          bool useDevice = false, int *blockSizes = NULL) {
  GpuTimer timer;
  timer.Start();

  if (useDevice == false) {
    printf("\nRadix sort by host\n");
    sortByHost(in, n, out, nBits);
  } else // use device
  {
    printf("\nRadix sort by device\n");
    sortByDevice(in, n, out, nBits, blockSizes);
  }

  timer.Stop();
  printf("Time: %.3f ms\n", timer.Elapsed());
}

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

void checkCorrectness(uint32_t *out, uint32_t *correctOut, int n) {
  for (int i = 0; i < n; i++) {
    if (out[i] != correctOut[i]) {
      printf("INCORRECT :(\n");
      return;
    }
  }
  printf("CORRECT :)\n");
}

void printArray(uint32_t *a, int n) {
  for (int i = 0; i < n; i++)
    printf("%i ", a[i]);
  printf("\n");
}

void checkHostCorrectness(uint32_t *out, int n) {
  for (int i = 0; i < n - 1; i += 1) {
    if (out[i] > out[i + 1]) {
      printf("INCORRECT :(\n");
      return;
    }
  }
  printf("CORRECT :)\n");
}

int main(int argc, char **argv) {
  // PRINT OUT DEVICE INFO
  printDeviceInfo();

  // SET UP INPUT SIZE
  int n = (1 << 24) + 1;
  // n = 10;
  printf("\nInput size: %d\n", n);

  // ALLOCATE MEMORIES
  size_t bytes = n * sizeof(uint32_t);
  uint32_t *in = (uint32_t *)malloc(bytes);
  uint32_t *out = (uint32_t *)malloc(bytes);        // Device result
  uint32_t *correctOut = (uint32_t *)malloc(bytes); // Host result

  // SET UP INPUT DATA
  for (int i = 0; i < n; i++)
    in[i] = rand();
  // printArray(in, n);

  // SET UP NBITS
  int nBits = 4; // Default
  if (argc > 1)
    nBits = atoi(argv[1]);
  printf("\nNum bits per digit: %d\n", nBits);

  // DETERMINE BLOCK SIZES
  int blockSizes[4] = {512, 512, 512,
                       512}; // One for histogram, one for scan, one for
                             // convert, one for couting sort
  if (argc == 6) {
    blockSizes[0] = atoi(argv[2]);
    blockSizes[1] = atoi(argv[3]);
    blockSizes[2] = atoi(argv[4]);
    blockSizes[3] = atoi(argv[5]);
  }
  printf("\nHist block size: %d, scan block size: %d, convert block size: %d, "
         "couting sort block size: %d\n",
         blockSizes[0], blockSizes[1], blockSizes[2], blockSizes[3]);

  // SORT BY HOST
  sort(in, n, correctOut, nBits);
  // printArray(correctOut, n);
  checkHostCorrectness(correctOut, n);

  // SORT BY DEVICE
  sort(in, n, out, nBits, true, blockSizes);
  checkCorrectness(out, correctOut, n);

  // FREE MEMORIES
  free(in);
  free(out);
  free(correctOut);

  return EXIT_SUCCESS;
}