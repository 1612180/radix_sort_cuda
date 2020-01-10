#include "sort.h"

void sortBase4(uint32_t *in, int n, uint32_t *out, int nBits, int *blockSizes) {
  int nBins = 1 << nBits; // 2^nBits

  uint32_t *tempIn = (uint32_t *)malloc(n * sizeof(uint32_t));
  uint32_t *tempOut = (uint32_t *)malloc(n * sizeof(uint32_t));

  memcpy(tempIn, in, n * sizeof(uint32_t));

  int histogramBlockSize = blockSizes[0];
  int histogramGridSize = (n - 1) / histogramBlockSize + 1;
  int histSize = histogramGridSize * nBins;
  uint32_t *hist = (uint32_t *)malloc(histSize * sizeof(uint32_t));
  uint32_t *histScanExclusive = (uint32_t *)malloc(histSize * sizeof(uint32_t));

  for (int bitBig = 0; bitBig < sizeof(uint32_t) * 8; bitBig += nBits) {
    // tempIn -> hist
    memset(hist, 0, histSize * sizeof(uint32_t));
    for (int i = 0; i < n; i += 1) {
      int bin = (tempIn[i] >> bitBig) & (nBins - 1);
      hist[bin * histogramGridSize + i / histogramBlockSize] += 1;
    }

    // hist -> histScanExclusive
    histScanExclusive[0] = 0;
    for (int i = 1; i < histSize; i += 1) {
      histScanExclusive[i] = histScanExclusive[i - 1] + hist[i - 1];
    }

    for (int i = 0; i < n; i += 1) {
      int bin = (tempIn[i] >> bitBig) & (nBins - 1);
      int rank = bin * histogramGridSize + i / histogramBlockSize;
      tempOut[histScanExclusive[rank]] = tempIn[i];
      histScanExclusive[rank] += 1;
    }

    uint32_t *temp = tempIn;
    tempIn = tempOut;
    tempOut = temp;
  }

  memcpy(out, tempIn, n * sizeof(uint32_t));

  free(tempIn);
  free(tempOut);
  free(hist);
  free(histScanExclusive);
}