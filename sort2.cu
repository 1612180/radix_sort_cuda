#include "sort.h"

// in -> hist
__global__ void computeHistKernel2(uint32_t *in, int n, uint32_t *hist,
                                   int nBins, int bit) {
  extern __shared__ uint32_t sHist[];
  // current threadIdx control threadIdx.x(+..blockDim.x) index in sHist
  for (int bin = threadIdx.x; bin < nBins; bin += blockDim.x) {
    sHist[bin] = 0;
  }
  __syncthreads();

  // local hist in block
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // >> de xoa cac bit o ben phai khong o vi tri "bit"
    // & de xoa cac bit khong o vi tri "bit"
    int bin = (in[i] >> bit) & (nBins - 1);
    atomicAdd(&sHist[bin], 1);
  }
  __syncthreads();

  // add local hist to global hist
  for (int bin = threadIdx.x; bin < nBins; bin += blockDim.x) {
    atomicAdd(&hist[bin], sHist[bin]);
  }
}

// in -> out, blockSums
__global__ void scanBlockKernel2(uint32_t *in, int n, uint32_t *out,
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

__global__ void addScannedBlockSums2(uint32_t *out, int n,
                                     uint32_t *blockSums) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  // skip first block index
  if (i < n && blockIdx.x > 0) {
    out[i] += blockSums[blockIdx.x - 1];
  }
}

__global__ void inclusiveToExclusive2(uint32_t *in, int n, uint32_t *inclusive,
                                      uint32_t *exclusive) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    exclusive[i] = inclusive[i] - in[i];
  }
}

void sortBase2(uint32_t *in, int n, uint32_t *out, int nBits, int *blockSizes) {
  int nBins = 1 << nBits; // 2^nBits

  uint32_t *tempIn = (uint32_t *)malloc(n * sizeof(uint32_t));
  uint32_t *tempOut = (uint32_t *)malloc(n * sizeof(uint32_t));
  memcpy(tempIn, in, n * sizeof(uint32_t));

  uint32_t *d_in;
  CHECK(cudaMalloc(&d_in, n * sizeof(uint32_t)));

  // compute historgram kernel
  int histogramBlockSize = blockSizes[0];
  dim3 histogramGridSize((n - 1) / histogramBlockSize + 1);

  uint32_t *d_hist;
  CHECK(cudaMalloc(&d_hist, nBins * sizeof(uint32_t)));

  // scan histogram kernel
  int scanBlockSize = blockSizes[1];
  dim3 scanGridSize((nBins - 1) / scanBlockSize + 1);

  uint32_t *blockSums = (uint32_t *)malloc(scanGridSize.x * sizeof(uint32_t));
  uint32_t *histScanExclusive = (uint32_t *)malloc(nBins * sizeof(uint32_t));

  uint32_t *d_histScan, *d_blockSums, *d_histScanExclusive;
  CHECK(cudaMalloc(&d_histScan, nBins * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_blockSums, scanGridSize.x * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_histScanExclusive, nBins * sizeof(uint32_t)));

  for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits) {
    CHECK(
        cudaMemcpy(d_in, tempIn, n * sizeof(uint32_t), cudaMemcpyHostToDevice))

    // d_in -> d_hist
    CHECK(cudaMemset(d_hist, 0, nBins * sizeof(uint32_t)))
    computeHistKernel2<<<histogramGridSize, histogramBlockSize,
                         nBins * sizeof(uint32_t)>>>(d_in, n, d_hist, nBins,
                                                     bit);
    CHECK(cudaDeviceSynchronize());

    // d_hist -> d_histScan
    scanBlockKernel2<<<scanGridSize, scanBlockSize,
                       scanBlockSize * sizeof(uint32_t)>>>(
        d_hist, nBins, d_histScan, d_blockSums);
    CHECK(cudaDeviceSynchronize());

    // scan d_blockSums
    CHECK(cudaMemcpy(blockSums, d_blockSums, scanGridSize.x * sizeof(uint32_t),
                     cudaMemcpyDeviceToHost));
    for (int i = 1; i < scanGridSize.x; i += 1) {
      blockSums[i] += blockSums[i - 1];
    }
    CHECK(cudaMemcpy(d_blockSums, blockSums, scanGridSize.x * sizeof(uint32_t),
                     cudaMemcpyHostToDevice));

    // d_histScan + d_blockSums
    addScannedBlockSums2<<<scanGridSize, scanBlockSize>>>(d_histScan, nBins,
                                                          d_blockSums);
    CHECK(cudaDeviceSynchronize());

    // d_hist, d_histScan -> d_histScanExclusive
    inclusiveToExclusive2<<<scanGridSize, scanBlockSize>>>(
        d_hist, nBins, d_histScan, d_histScanExclusive);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(histScanExclusive, d_histScanExclusive,
                     nBins * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // histScanExclusive -> tempOut
    for (int i = 0; i < n; i += 1) {
      int bin = (tempIn[i] >> bit) & (nBins - 1);
      tempOut[histScanExclusive[bin]] = tempIn[i];
      histScanExclusive[bin] += 1;
    }

    uint32_t *temp = tempIn;
    tempIn = tempOut;
    tempOut = temp;
  }

  memcpy(out, tempIn, n * sizeof(uint32_t));
}