#include "sort.h"

void sortBase1(uint32_t *in, int n, uint32_t *out, int nBits) {
  // use for histogram
  int nBins = 1 << nBits; // 2^nBits
  uint32_t *hist = (uint32_t *)malloc(nBins * sizeof(uint32_t));
  uint32_t *histScan = (uint32_t *)malloc(nBins * sizeof(uint32_t));

  // tempIn, tempOut to replace for in, out when loop radix sort
  uint32_t *tempIn = (uint32_t *)malloc(n * sizeof(uint32_t));
  uint32_t *tempOut = (uint32_t *)malloc(n * sizeof(uint32_t));
  memcpy(tempIn, in, n * sizeof(uint32_t));

  // loop from most smallest digit -> most biggest digit
  // 1 byte = 8 bits
  for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits) {
    // tempIn -> hist
    memset(hist, 0, nBins * sizeof(int));
    for (int i = 0; i < n; i += 1) {
      int bin = (tempIn[i] >> bit) & (nBins - 1);
      hist[bin]++;
    }

    // hist -> histScan
    histScan[0] = 0;
    for (int bin = 1; bin < nBins; bin += 1) {
      histScan[bin] = histScan[bin - 1] + hist[bin - 1];
    }

    // histScan -> tempOut
    for (int i = 0; i < n; i++) {
      int bin = (tempIn[i] >> bit) & (nBins - 1);
      tempOut[histScan[bin]] = tempIn[i];
      histScan[bin] += 1;
    }

    uint32_t *temp = tempIn;
    tempIn = tempOut;
    tempOut = temp;
  }

  memcpy(out, tempIn, n * sizeof(uint32_t));

  free(hist);
  free(histScan);
  free(tempIn);
  free(tempOut);
}
