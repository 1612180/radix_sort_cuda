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

void printDebug(uint32_t *arr, int n, const char *debugMessage) {
  printf("%s\n", debugMessage);
  for (int i = 0; i < n; i += 1) {
    printf("%d ", arr[i]);
  }
  printf("\n");
}

//   0 1 2 ... gridDim.x - 1
// 0
// 1
// 2
// ...
// nBins - 1
// hist: 2D array
// convert to 1D array
// (0 1 2 ... gridDim.x - 1)  (of 0) ... (0 1 2 ... gridDim.x - 1) (of nBins -
// 1)

__global__ void computeHist2DKernel(const uint32_t *in, int n, uint32_t *hist,
                                    int nBins, int bitBig) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int bin = (in[i] >> bitBig) & (nBins - 1);
    atomicAdd(&hist[bin * gridDim.x + blockIdx.x], 1);
  }
}

// scan on in -> out, blockSums
__global__ void scanBlockKernel(uint32_t *in, int n, uint32_t *out,
                                uint32_t *blockSums) {
  // TODO
  extern __shared__ uint32_t section[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    section[threadIdx.x] = in[i];
  } else {
    section[threadIdx.x] = 0;
  }
  __syncthreads();

  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    // copy section[threadIdx.x - stride] before changed
    int previous = 0;
    if (stride <= threadIdx.x) {
      previous = section[threadIdx.x - stride];
    }
    __syncthreads();

    section[threadIdx.x] += previous;
    __syncthreads();
  }

  if (i < n) {
    out[i] = section[threadIdx.x];
  }

  if (blockSums != NULL && threadIdx.x == 0) {
    blockSums[blockIdx.x] = section[blockDim.x - 1];
  }
}

// add blockSums
__global__ void addScannedBlockSums(uint32_t *out, int n, uint32_t *blockSums) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  // skip first block index in "out"
  if (i < n && blockIdx.x > 0) {
    out[i] += blockSums[blockIdx.x - 1];
  }
}

__global__ void inclusiveToExclusive(const uint32_t *in, int n,
                                     uint32_t *inclusive, uint32_t *exclusive) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    exclusive[i] = inclusive[i] - in[i];
  }
}

__global__ void convertBinary(const uint32_t *in, int n, uint32_t *inBinary,
                              int nBins, int bitBig, int bitSmall) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int temp = (in[i] >> bitBig) & (nBins - 1);
    inBinary[i] = (temp >> bitSmall) & 1;
  }
}

__global__ void calculateRankPerBlock(const uint32_t *inBinary, int n,
                                      uint32_t *inBinaryScan,
                                      uint32_t *nZerosPerBlock,
                                      uint32_t *inRankPerBlock) {
  // calculate nZeros for each block
  int lastBlock = blockIdx.x * blockDim.x + blockDim.x - 1;
  int countBlock = blockDim.x;

  // last block is bigger than remain elements
  // example
  // n = 3 => 0 1 2
  // gridSize: 2 -> 2 blocks
  // blockSize: 2 -> 2 threads
  // (0 1) (2)
  // last block only contain (2)
  //
  // count thread in last block is actually < blockDim.x

  if (lastBlock >= n) {
    lastBlock = n - 1;
    countBlock = n - (blockIdx.x * blockDim.x);
  }

  if (threadIdx.x == 0) {
    nZerosPerBlock[blockIdx.x] =
        countBlock - inBinaryScan[lastBlock] - inBinary[lastBlock];
  }
  __syncthreads();

  // calculate rank with nZeros
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    if (inBinary[i] == 0) {
      // threadIdx.x replace for i because we only calculate inside block
      inRankPerBlock[i] = threadIdx.x - inBinaryScan[i];
    } else if (inBinary[i] == 1) {
      inRankPerBlock[i] = nZerosPerBlock[blockIdx.x] + inBinaryScan[i];
    }
  }
}

__global__ void coutingSortPerBlock(const uint32_t *in, int n,
                                    uint32_t *inRankPerBlock, uint32_t *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // because inRankPerBlock[i] is only rank in[i] inside block
    // we need to plus the previous block
    int rank = blockIdx.x * blockDim.x + inRankPerBlock[i];
    out[rank] = in[i];
  }
}

// 1 1 2 2 3 3 4 4 4
// init 1
// 1 1 1 1 1 1 1 1 1
// stride = 1
// 1 2 1 2 1 2 1 2 2
// stride = 2
// 1 2 1 2 1 2 1 2 3
// decrease by 1
// 0 1 0 1 0 1 0 1 2
__global__ void scanBlockKernelWithEqual(const uint32_t *in, int n,
                                         uint32_t *outWithEqual, int nBins,
                                         int bitBig) {
  extern __shared__ uint32_t section[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // copy in to SMEM
  if (i < n) {
    section[threadIdx.x] = 1;
  }
  __syncthreads();

  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    // copy previous section[threadIdx.x - stride] before changed
    int previous = 0;
    if (stride <= threadIdx.x) {
      previous = section[threadIdx.x - stride];
    }
    __syncthreads();

    // only + if equal
    if (stride <= threadIdx.x && i < n) {
      int bin = (in[i] >> bitBig) & (nBins - 1);
      int binPrevious = (in[i - stride] >> bitBig) & (nBins - 1);
      if (bin == binPrevious) {
        section[threadIdx.x] += previous;
      }
    }
    __syncthreads();
  }

  __syncthreads();
  if (i < n) {
    outWithEqual[i] = section[threadIdx.x] - 1;
  }
}

__global__ void coutingSortWithHist(const uint32_t *in, int n, uint32_t *out,
                                    uint32_t *histScan, int nBins, int bitBig,
                                    uint32_t *outWithEqual) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int bin = (in[i] >> bitBig) & (nBins - 1);
    int rankHist = histScan[bin * gridDim.x + blockIdx.x];

    int rankPerBlock = outWithEqual[i];

    out[rankHist + rankPerBlock] = in[i];
  }
}

void sortByDevice(const uint32_t *in, int n, uint32_t *out, int nBits,
                  int *blockSizes) {
  int nBins = 1 << nBits; // 2^nBits
  printf("nBins %d\n", nBins);

  uint32_t *tempN = (uint32_t *)malloc(n * sizeof(uint32_t));

  uint32_t *d_in, *d_out;
  CHECK(cudaMalloc(&d_in, n * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_out, n * sizeof(uint32_t)));

  CHECK(cudaMemcpy(d_in, in, n * sizeof(uint32_t), cudaMemcpyHostToDevice));

  // histogram kernel
  int histogramBlockSize = blockSizes[0];
  dim3 histogramGridSize((n - 1) / histogramBlockSize + 1);

  int histSize = histogramGridSize.x * nBins;
  printf("histSize %d\n", histSize);

  uint32_t *hist = (uint32_t *)malloc(histSize * sizeof(uint32_t));

  uint32_t *d_hist;
  CHECK(cudaMalloc(&d_hist, histSize * sizeof(uint32_t)));

  // scan histogram kernel
  int scanHistogramBlockSize = blockSizes[1];
  dim3 scanHistogramGridSize((histSize - 1) / scanHistogramBlockSize + 1);

  uint32_t *histScan = (uint32_t *)malloc(histSize * sizeof(uint32_t));
  uint32_t *blockSums =
      (uint32_t *)malloc(scanHistogramGridSize.x * sizeof(uint32_t));

  uint32_t *d_histScan, *d_blockSums;
  CHECK(cudaMalloc(&d_histScan, histSize * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_blockSums, scanHistogramGridSize.x * sizeof(uint32_t)));

  uint32_t *histScanExclusive = (uint32_t *)malloc(histSize * sizeof(uint32_t));

  uint32_t *d_histScanExclusive;
  CHECK(cudaMalloc(&d_histScanExclusive, histSize * sizeof(uint32_t)));

  // other kernel
  int otherBlockSize = blockSizes[2];
  dim3 otherGridSize((n - 1) / otherBlockSize + 1);

  uint32_t *inBinary = (uint32_t *)malloc(n * sizeof(uint32_t));

  uint32_t *d_inBinary;
  CHECK(cudaMalloc(&d_inBinary, n * sizeof(uint32_t)));

  uint32_t *inBinaryScan = (uint32_t *)malloc(n * sizeof(uint32_t));

  uint32_t *d_inBinaryScan;
  CHECK(cudaMalloc(&d_inBinaryScan, n * sizeof(uint32_t)));

  uint32_t *inBinaryScanExclusive = (uint32_t *)malloc(n * sizeof(uint32_t));

  uint32_t *d_inBinaryScanExclusive;
  CHECK(cudaMalloc(&d_inBinaryScanExclusive, n * sizeof(uint32_t)));

  uint32_t *inRankPerBlock = (uint32_t *)malloc(n * sizeof(uint32_t));
  uint32_t *nZerosPerBlock =
      (uint32_t *)malloc(otherGridSize.x * sizeof(uint32_t));

  uint32_t *d_inRankPerBlock, *d_nZerosPerBlock;
  CHECK(cudaMalloc(&d_inRankPerBlock, n * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_nZerosPerBlock, otherGridSize.x * sizeof(uint32_t)));

  uint32_t *outWithEqual = (uint32_t *)malloc(n * sizeof(uint32_t));

  uint32_t *d_outWithEqual;
  CHECK(cudaMalloc(&d_outWithEqual, n * sizeof(uint32_t)));

  // 1 byte = 8 bits
  for (int bitBig = 0; bitBig < sizeof(uint32_t) * 8; bitBig += nBits) {
    // for (int bitBig = 0; bitBig < nBits; bitBig += nBits) {
    // CHECK(
    //     cudaMemcpy(tempN, d_in, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    // printDebug(tempN, n, "d_in");

    // d_in -> d_hist
    CHECK(cudaMemset(d_hist, 0, histSize * sizeof(uint32_t)));
    computeHist2DKernel<<<histogramGridSize, histogramBlockSize>>>(
        d_in, n, d_hist, nBins, bitBig);
    CHECK(cudaDeviceSynchronize());

    // CHECK(cudaMemcpy(hist, d_hist, histSize * sizeof(uint32_t),
    //                  cudaMemcpyDeviceToHost));
    // printDebug(hist, histSize, "d_hist");

    // d_hist -> d_histScan
    scanBlockKernel<<<scanHistogramGridSize, scanHistogramBlockSize,
                      scanHistogramBlockSize * sizeof(uint32_t)>>>(
        d_hist, histSize, d_histScan, d_blockSums);
    CHECK(cudaDeviceSynchronize());

    // d_blockSums -> blockSums
    // inclusive scan blockSums on host
    // blockSums -> d_blockSums
    CHECK(cudaMemcpy(blockSums, d_blockSums,
                     scanHistogramGridSize.x * sizeof(uint32_t),
                     cudaMemcpyDeviceToHost));
    // printDebug(blockSums, scanHistogramGridSize.x, "blockSums pre");
    for (int i = 1; i < scanHistogramGridSize.x; i += 1) {
      blockSums[i] += blockSums[i - 1];
    }
    // printDebug(blockSums, scanHistogramGridSize.x, "blockSums after");
    CHECK(cudaMemcpy(d_blockSums, blockSums,
                     scanHistogramGridSize.x * sizeof(uint32_t),
                     cudaMemcpyHostToDevice));

    // add d_blockSums to d_histScan
    addScannedBlockSums<<<scanHistogramGridSize, scanHistogramBlockSize>>>(
        d_histScan, histSize, d_blockSums);
    CHECK(cudaDeviceSynchronize());

    // CHECK(cudaMemcpy(histScan, d_histScan, histSize * sizeof(uint32_t),
    //                  cudaMemcpyDeviceToHost));
    // printDebug(histScan, histSize, "d_histScan");

    // d_hist, d_histScan -> d_histScanExclusive
    inclusiveToExclusive<<<scanHistogramGridSize, scanHistogramBlockSize>>>(
        d_hist, histSize, d_histScan, d_histScanExclusive);
    CHECK(cudaDeviceSynchronize());

    // CHECK(cudaMemcpy(histScanExclusive, d_histScanExclusive,
    //                  histSize * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    // printDebug(histScanExclusive, histSize, "d_histScanExclusive");

    // radix sort each block
    for (int bitSmall = 0; bitSmall < nBits; bitSmall += 1) {
      // printf("bitSmall %d\n", bitSmall);

      // d_in -> d_inBinary
      convertBinary<<<otherGridSize, otherBlockSize>>>(d_in, n, d_inBinary,
                                                       nBins, bitBig, bitSmall);
      CHECK(cudaDeviceSynchronize());

      // CHECK(cudaMemcpy(inBinary, d_inBinary, n * sizeof(uint32_t),
      //                  cudaMemcpyDeviceToHost));
      // printDebug(inBinary, n, "d_inBinary");

      // d_inBinary -> d_inBinaryScan
      scanBlockKernel<<<otherGridSize, otherBlockSize,
                        otherBlockSize * sizeof(uint32_t)>>>(
          d_inBinary, n, d_inBinaryScan, NULL);
      CHECK(cudaDeviceSynchronize());

      // CHECK(cudaMemcpy(inBinaryScan, d_inBinaryScan, n * sizeof(uint32_t),
      //                  cudaMemcpyDeviceToHost));
      // printDebug(inBinaryScan, n, "d_inBinaryScan");

      // d_inBinaryScan -> d_inBinaryScanExclusive
      inclusiveToExclusive<<<otherGridSize, otherBlockSize>>>(
          d_inBinary, n, d_inBinaryScan, d_inBinaryScanExclusive);
      CHECK(cudaDeviceSynchronize());

      // CHECK(cudaMemcpy(inBinaryScanExclusive, d_inBinaryScanExclusive,
      //                  n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
      // printDebug(inBinaryScanExclusive, n, "d_inBinaryScanExclusive");

      // d_inBinary, d_inBinaryScanExclusive -> d_nZerosPerBlock,
      // d_inRankPerBlock
      calculateRankPerBlock<<<otherGridSize, otherBlockSize>>>(
          d_inBinary, n, d_inBinaryScanExclusive, d_nZerosPerBlock,
          d_inRankPerBlock);
      CHECK(cudaDeviceSynchronize());

      CHECK(cudaMemcpy(nZerosPerBlock, d_nZerosPerBlock,
                       otherGridSize.x * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));
      // printDebug(nZerosPerBlock, n, "d_nZerosPerBlock");

      // CHECK(cudaMemcpy(inRankPerBlock, d_inRankPerBlock, n * sizeof(uint32_t),
      //                  cudaMemcpyDeviceToHost));
      // printDebug(inRankPerBlock, n, "d_inRankPerBlock");

      // d_in, d_inRankPerBlock -> d_out
      coutingSortPerBlock<<<otherGridSize, otherBlockSize>>>(
          d_in, n, d_inRankPerBlock, d_out);
      CHECK(cudaDeviceSynchronize());

      uint32_t *temp = d_in;
      d_in = d_out;
      d_out = temp;
    }

    scanBlockKernelWithEqual<<<otherGridSize, otherBlockSize,
                               otherBlockSize * sizeof(uint32_t)>>>(
        d_in, n, d_outWithEqual, nBins, bitBig);
    CHECK(cudaDeviceSynchronize());

    // CHECK(cudaMemcpy(outWithEqual, d_outWithEqual, n * sizeof(uint32_t),
    //                  cudaMemcpyDeviceToHost));
    // printDebug(outWithEqual, n, "d_outWithEqual");

    coutingSortWithHist<<<histogramGridSize, histogramBlockSize>>>(
        d_in, n, d_out, d_histScanExclusive, nBins, bitBig, d_outWithEqual);
    CHECK(cudaDeviceSynchronize());

    // CHECK(
    //     cudaMemcpy(tempN, d_out, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    // printDebug(tempN, n, "d_out");

    uint32_t *temp = d_in;
    d_in = d_out;
    d_out = temp;
  }

  CHECK(cudaMemcpy(out, d_in, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
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

int main(int argc, char **argv) {
  // PRINT OUT DEVICE INFO
  printDeviceInfo();

  // SET UP INPUT SIZE
  int n = (1 << 24) + 1;
  // n = 1000000;
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
  int blockSizes[3] = {512, 512, 512}; // One for histogram, one for scan, one
                                       // for other
  if (argc == 5) {
    blockSizes[0] = atoi(argv[2]);
    blockSizes[1] = atoi(argv[3]);
    blockSizes[2] = atoi(argv[4]);
  }
  printf("\nHist block size: %d, scan block size: %d\n", blockSizes[0],
         blockSizes[1]);

  // SORT BY HOST
  sort(in, n, correctOut, nBits);
  // printArray(correctOut, n);

  // SORT BY DEVICE
  sort(in, n, out, nBits, true, blockSizes);
  checkCorrectness(out, correctOut, n);

  // FREE MEMORIES
  free(in);
  free(out);
  free(correctOut);

  return EXIT_SUCCESS;
}
