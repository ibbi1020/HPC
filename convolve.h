/*********************************************************************
 * convolve.h
 *********************************************************************/

#ifndef _CONVOLVE_H_
#define _CONVOLVE_H_

#include "klt.h"
#include "klt_util.h"

#define MAX_KERNEL_WIDTH 71

typedef struct
{
  int width;
  float data[MAX_KERNEL_WIDTH];
} ConvolutionKernel;

/* Get internal Gaussian kernel pointers (for pyramid optimization) */
void _KLTGetGaussianKernels(ConvolutionKernel **gauss, ConvolutionKernel **gaussderiv);

void _KLTToFloatImage(
  KLT_PixelType *img,
  int ncols, int nrows,
  _KLT_FloatImage floatimg);

void _KLTComputeGradients(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage gradx,
  _KLT_FloatImage grady);

void _KLTGetKernelWidths(
  float sigma,
  int *gauss_width,
  int *gaussderiv_width);

void _KLTComputeSmoothedImage(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage smooth);

#endif
