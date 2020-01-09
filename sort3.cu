#include "sort.h"

// in -> out, blockSums
__global__ void scanBlockKernel3(uint32_t *in, int n, uint32_t *out,
                                 uint32_t *blockSums) {
  extern __shared__ uint32_t section[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    section[threadIdx.x] = in[i];
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

  __syncthreads();
  if (i < n) {
    out[i] = section[threadIdx.x];
  }

  if (blockSums != NULL && threadIdx.x == 0) {
    blockSums[blockIdx.x] = section[blockDim.x - 1];
  }
}

__global__ void addScannedBlockSums3(uint32_t *out, int n,
                                     uint32_t *blockSums) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  // skip first block index
  if (i < n && blockIdx.x > 0) {
    out[i] += blockSums[blockIdx.x - 1];
  }
}

__global__ void inclusiveToExclusive3(uint32_t *in, int n, uint32_t *inclusive,
                                      uint32_t *exclusive) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    exclusive[i] = inclusive[i] - in[i];
  }
}

__global__ void convertBinary3(uint32_t *in, int n, uint32_t *inBinary,
                               int bit) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    inBinary[i] = (in[i] >> bit) & 1;
  }
}

__global__ void countingSort3(uint32_t *in, int n, uint32_t *out,
                              uint32_t *inBinary, uint32_t *inScanExclusive,
                              int nZeros) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    int rank =
        inBinary[i] == 0 ? i - inScanExclusive[i] : nZeros + inScanExclusive[i];
    out[rank] = in[i];
  }
}

void sortBase3(uint32_t *in, int n, uint32_t *out, int nBits, int *blockSizes) {
  uint32_t *d_in, *d_out;
  CHECK(cudaMalloc(&d_in, n * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_out, n * sizeof(uint32_t)));
  CHECK(cudaMemcpy(d_in, in, n * sizeof(uint32_t), cudaMemcpyHostToDevice));

  // scan kernel
  int scanBlockSize = blockSizes[0];
  dim3 scanGridSize((n - 1) / scanBlockSize + 1);

  uint32_t *blockSums = (uint32_t *)malloc(scanGridSize.x * sizeof(uint32_t));
  uint32_t *inScanExclusive = (uint32_t *)malloc(n * sizeof(uint32_t));

  uint32_t *d_inScan, *d_blockSums, *d_inScanExclusive;
  CHECK(cudaMalloc(&d_inScan, n * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_blockSums, scanGridSize.x * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_inScanExclusive, n * sizeof(uint32_t)));

  int otherBlockSize = blockSizes[1];
  dim3 otherGridSize((n - 1) / otherBlockSize + 1);

  // convert kernel
  uint32_t *inBinary = (uint32_t *)malloc(n * sizeof(uint32_t));

  uint32_t *d_inBinary;
  CHECK(cudaMalloc(&d_inBinary, n * sizeof(uint32_t)));

  for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += 1) {
    // d_in -> d_inBinary
    convertBinary3<<<otherGridSize, otherBlockSize>>>(d_in, n, d_inBinary, bit);
    CHECK(cudaDeviceSynchronize());

    // d_inBinary -> d_inScan
    scanBlockKernel3<<<scanGridSize, scanBlockSize,
                       scanBlockSize * sizeof(uint32_t)>>>(
        d_inBinary, n, d_inScan, d_blockSums);
    CHECK(cudaDeviceSynchronize());

    // scan d_blockSums
    CHECK(cudaMemcpy(blockSums, d_blockSums, scanGridSize.x * sizeof(uint32_t),
                     cudaMemcpyDeviceToHost));
    for (int i = 1; i < scanGridSize.x; i += 1) {
      blockSums[i] += blockSums[i - 1];
    }
    CHECK(cudaMemcpy(d_blockSums, blockSums, scanGridSize.x * sizeof(uint32_t),
                     cudaMemcpyHostToDevice));

    // d_inScan + d_blockSums
    addScannedBlockSums3<<<scanGridSize, scanBlockSize>>>(d_inScan, n,
                                                          d_blockSums);
    CHECK(cudaDeviceSynchronize());

    // d_inScan -> d_inScanExclusive
    inclusiveToExclusive3<<<otherGridSize, otherBlockSize>>>(
        d_inBinary, n, d_inScan, d_inScanExclusive);
    CHECK(cudaDeviceSynchronize());

    // d_inScanExclusive, d_inBinary -> nZeros
    CHECK(cudaMemcpy(inScanExclusive, d_inScanExclusive, n * sizeof(uint32_t),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(inBinary, d_inBinary, n * sizeof(uint32_t),
                     cudaMemcpyDeviceToHost));
    int nZeros = n - inScanExclusive[n - 1] - inBinary[n - 1];

    countingSort3<<<otherGridSize, otherBlockSize>>>(d_in, n, d_out, d_inBinary,
                                                    d_inScanExclusive, nZeros);
    CHECK(cudaDeviceSynchronize());

    uint32_t *temp = d_in;
    d_in = d_out;
    d_out = temp;
  }

  free(blockSums);
  free(inScanExclusive);
  free(inBinary);

  cudaFree(d_inScan);
  cudaFree(d_blockSums);
  cudaFree(d_inScanExclusive);
  cudaFree(d_inBinary);
}