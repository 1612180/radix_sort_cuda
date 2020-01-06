#include "sort.h"

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
__global__ void computeHist2DKernel41(uint32_t *in, int n, uint32_t *hist,
                                      int nBins, int bitBig) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int bin = (in[i] >> bitBig) & (nBins - 1);
    atomicAdd(&hist[bin * gridDim.x + blockIdx.x], 1);
  }
}

// in -> out, blockSums
__global__ void scanBlockKernel41(uint32_t *in, int n, uint32_t *out,
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

__global__ void addScannedBlockSums41(uint32_t *out, int n,
                                      uint32_t *blockSums) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  // skip first block index
  if (i < n && blockIdx.x > 0) {
    out[i] += blockSums[blockIdx.x - 1];
  }
}

__global__ void inclusiveToExclusive41(uint32_t *in, int n, uint32_t *inclusive,
                                       uint32_t *exclusive) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    exclusive[i] = inclusive[i] - in[i];
  }
}

__global__ void convertBinary41(uint32_t *in, int n, uint32_t *inBinary,
                                int nBins, int bitBig, int bitSmall) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int temp = (in[i] >> bitBig) & (nBins - 1);
    inBinary[i] = (temp >> bitSmall) & 1;
  }
}

__global__ void calculateRankPerBlock41(uint32_t *inBinary, int n,
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

__global__ void coutingSortPerBlock41(uint32_t *in, int n,
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
__global__ void scanBlockKernelWithEqual41(uint32_t *in, int n,
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

__global__ void coutingSortWithHist41(uint32_t *in, int n, uint32_t *out,
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

void sortBase41(uint32_t *in, int n, uint32_t *out, int nBits,
                int *blockSizes) {
  int nBins = 1 << nBits; // 2^nBits

  uint32_t *tempN = (uint32_t *)malloc(n * sizeof(uint32_t));

  uint32_t *d_in, *d_out;
  CHECK(cudaMalloc(&d_in, n * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_out, n * sizeof(uint32_t)));
  CHECK(cudaMemcpy(d_in, in, n * sizeof(uint32_t), cudaMemcpyHostToDevice));

  // histogram kernel
  int histogramBlockSize = blockSizes[0];
  dim3 histogramGridSize((n - 1) / histogramBlockSize + 1);

  int histSize = histogramGridSize.x * nBins;

  uint32_t *d_hist;
  CHECK(cudaMalloc(&d_hist, histSize * sizeof(uint32_t)));

  // scan histogram kernel
  int scanHistogramBlockSize = blockSizes[1];
  dim3 scanHistogramGridSize((histSize - 1) / scanHistogramBlockSize + 1);

  uint32_t *blockSums =
      (uint32_t *)malloc(scanHistogramGridSize.x * sizeof(uint32_t));

  uint32_t *d_histScan, *d_blockSums, *d_histScanExclusive;
  CHECK(cudaMalloc(&d_histScan, histSize * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_blockSums, scanHistogramGridSize.x * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_histScanExclusive, histSize * sizeof(uint32_t)));

  // other kernel
  int otherBlockSize = blockSizes[2];
  dim3 otherGridSize((n - 1) / otherBlockSize + 1);

  uint32_t *d_inBinary;
  CHECK(cudaMalloc(&d_inBinary, n * sizeof(uint32_t)));

  uint32_t *d_inBinaryScan, *d_inBinaryScanExclusive;
  CHECK(cudaMalloc(&d_inBinaryScan, n * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_inBinaryScanExclusive, n * sizeof(uint32_t)));

  uint32_t *d_inRankPerBlock, *d_nZerosPerBlock;
  CHECK(cudaMalloc(&d_inRankPerBlock, n * sizeof(uint32_t)));
  CHECK(cudaMalloc(&d_nZerosPerBlock, otherGridSize.x * sizeof(uint32_t)));

  uint32_t *d_outWithEqual;
  CHECK(cudaMalloc(&d_outWithEqual, n * sizeof(uint32_t)));

  for (int bitBig = 0; bitBig < sizeof(uint32_t) * 8; bitBig += nBits) {
    // d_in -> d_hist
    CHECK(cudaMemset(d_hist, 0, histSize * sizeof(uint32_t)));
    computeHist2DKernel41<<<histogramGridSize, histogramBlockSize>>>(
        d_in, n, d_hist, nBins, bitBig);
    CHECK(cudaDeviceSynchronize());

    // d_hist -> d_histScan
    scanBlockKernel41<<<scanHistogramGridSize, scanHistogramBlockSize,
                        scanHistogramBlockSize * sizeof(uint32_t)>>>(
        d_hist, histSize, d_histScan, d_blockSums);
    CHECK(cudaDeviceSynchronize());

    // scan d_blockSums
    CHECK(cudaMemcpy(blockSums, d_blockSums,
                     scanHistogramGridSize.x * sizeof(uint32_t),
                     cudaMemcpyDeviceToHost));
    for (int i = 1; i < scanHistogramGridSize.x; i += 1) {
      blockSums[i] += blockSums[i - 1];
    }
    CHECK(cudaMemcpy(d_blockSums, blockSums,
                     scanHistogramGridSize.x * sizeof(uint32_t),
                     cudaMemcpyHostToDevice));

    // d_histScan + d_blockSums
    addScannedBlockSums41<<<scanHistogramGridSize, scanHistogramBlockSize>>>(
        d_histScan, histSize, d_blockSums);
    CHECK(cudaDeviceSynchronize());

    // d_hist, d_histScan -> d_histScanExclusive
    inclusiveToExclusive41<<<scanHistogramGridSize, scanHistogramBlockSize>>>(
        d_hist, histSize, d_histScan, d_histScanExclusive);
    CHECK(cudaDeviceSynchronize());

    for (int bitSmall = 0; bitSmall < nBits; bitSmall += 1) {
      // d_in -> d_inBinary
      convertBinary41<<<otherGridSize, otherBlockSize>>>(d_in, n, d_inBinary,
                                                       nBins, bitBig, bitSmall);
      CHECK(cudaDeviceSynchronize());

      // d_inBinary -> d_inBinaryScan
      scanBlockKernel41<<<otherGridSize, otherBlockSize,
                          otherBlockSize * sizeof(uint32_t)>>>(
          d_inBinary, n, d_inBinaryScan, NULL);
      CHECK(cudaDeviceSynchronize());

      // d_inBinary, d_inBinaryScan -> d_inBinaryScanExclusive
      inclusiveToExclusive41<<<otherGridSize, otherBlockSize>>>(
          d_inBinary, n, d_inBinaryScan, d_inBinaryScanExclusive);
      CHECK(cudaDeviceSynchronize());

      // d_inBinary, d_inBinaryScanExclusive -> d_nZerosPerBlock,
      // d_inRankPerBlock
      calculateRankPerBlock41<<<otherGridSize, otherBlockSize>>>(
          d_inBinary, n, d_inBinaryScanExclusive, d_nZerosPerBlock,
          d_inRankPerBlock);
      CHECK(cudaDeviceSynchronize());

      coutingSortPerBlock41<<<otherGridSize, otherBlockSize>>>(
          d_in, n, d_inRankPerBlock, d_out);
      CHECK(cudaDeviceSynchronize());

      uint32_t *temp = d_in;
      d_in = d_out;
      d_out = temp;
    }

    scanBlockKernelWithEqual41<<<otherGridSize, otherBlockSize,
                                 otherBlockSize * sizeof(uint32_t)>>>(
        d_in, n, d_outWithEqual, nBins, bitBig);
    CHECK(cudaDeviceSynchronize());

    coutingSortWithHist41<<<histogramGridSize, histogramBlockSize>>>(
        d_in, n, d_out, d_histScanExclusive, nBins, bitBig, d_outWithEqual);
    CHECK(cudaDeviceSynchronize());

    uint32_t *temp = d_in;
    d_in = d_out;
    d_out = temp;
  }

  CHECK(cudaMemcpy(out, d_in, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}