/*********************************************************************
 * convolve.c
 *********************************************************************/


 #include <assert.h>
 #include <math.h>
 #include <stdlib.h> 
 
 #include "base.h"
 #include "error.h"
 #include "convolve.h"
 #include "klt_util.h" /* printing */
 //#include "cuda_config.h" /* CUDA optimization flags */
 
 
 /* Global tracking context for persistent buffer access */
 /* This is set by _KLTSetGlobalTrackingContext() before pyramid operations */
 KLT_TrackingContext _KLTGlobalTrackingContext = NULL;
 
 #define MAX_KERNEL_WIDTH 71
 
 typedef struct
 {
   int width;
   float data[MAX_KERNEL_WIDTH];
 } ConvolutionKernel;
 
 /* Kernels */
 static ConvolutionKernel gauss_kernel;
 static ConvolutionKernel gaussderiv_kernel;
 static float sigma_last = -10.0;
 
 /*********************************************************************
  * _KLTToFloatImage
  *
  * Given a pointer to image data (probably unsigned chars), copy
  * data to a float image.
  */
 
 void _KLTToFloatImage(
     KLT_PixelType *img,
     int ncols, int nrows,
     _KLT_FloatImage floatimg)
 {
   KLT_PixelType *ptrend = img + ncols * nrows;
   float *ptrout = floatimg->data;
 
   /* Output image must be large enough to hold result */
   assert(floatimg->ncols >= ncols);
   assert(floatimg->nrows >= nrows);
 
   floatimg->ncols = ncols;
   floatimg->nrows = nrows;
 
   while (img < ptrend)
     *ptrout++ = (float)*img++;
 }
 
 /*********************************************************************
  * _computeKernels
  */
 
 static void _computeKernels(
     float sigma,
     ConvolutionKernel *gauss,
     ConvolutionKernel *gaussderiv)
 {
   const float factor = 0.01f; /* for truncating tail */
   int i;
 
   assert(MAX_KERNEL_WIDTH % 2 == 1);
   assert(sigma >= 0.0);
 
   /* Compute kernels, and automatically determine widths */
   {
     const int hw = MAX_KERNEL_WIDTH / 2;
     float max_gauss = 1.0f, max_gaussderiv = (float)(sigma * exp(-0.5f));
 
     /* Compute gauss and deriv */
     for (i = -hw; i <= hw; i++)
     {
       gauss->data[i + hw] = (float)exp(-i * i / (2 * sigma * sigma));
       gaussderiv->data[i + hw] = -i * gauss->data[i + hw];
     }
 
     /* Compute widths */
     gauss->width = MAX_KERNEL_WIDTH;
     for (i = -hw; fabs(gauss->data[i + hw] / max_gauss) < factor;
          i++, gauss->width -= 2)
       ;
     gaussderiv->width = MAX_KERNEL_WIDTH;
     for (i = -hw; fabs(gaussderiv->data[i + hw] / max_gaussderiv) < factor;
          i++, gaussderiv->width -= 2)
       ;
     if (gauss->width == MAX_KERNEL_WIDTH ||
         gaussderiv->width == MAX_KERNEL_WIDTH)
       KLTError("(_computeKernels) MAX_KERNEL_WIDTH %d is too small for "
                "a sigma of %f",
                MAX_KERNEL_WIDTH, sigma);
   }
 
   /* Shift if width less than MAX_KERNEL_WIDTH */
   for (i = 0; i < gauss->width; i++)
     gauss->data[i] = gauss->data[i + (MAX_KERNEL_WIDTH - gauss->width) / 2];
   for (i = 0; i < gaussderiv->width; i++)
     gaussderiv->data[i] = gaussderiv->data[i + (MAX_KERNEL_WIDTH - gaussderiv->width) / 2];
   /* Normalize gauss and deriv */
   {
     const int hw = gaussderiv->width / 2;
     float den;
 
     den = 0.0;
     for (i = 0; i < gauss->width; i++)
       den += gauss->data[i];
     for (i = 0; i < gauss->width; i++)
       gauss->data[i] /= den;
     den = 0.0;
     for (i = -hw; i <= hw; i++)
       den -= i * gaussderiv->data[i + hw];
     for (i = -hw; i <= hw; i++)
       gaussderiv->data[i + hw] /= den;
   }
 
   sigma_last = sigma;
 }
 
 /*********************************************************************
  * _KLTGetKernelWidths
  *
  */
 
 void _KLTGetKernelWidths(
     float sigma,
     int *gauss_width,
     int *gaussderiv_width)
 {
   _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
   *gauss_width = gauss_kernel.width;
   *gaussderiv_width = gaussderiv_kernel.width;
 }
 
 /*********************************************************************
  * _convolveImageHoriz
  */
 
  static void _convolveImageHoriz(
    _KLT_FloatImage imgin,
    ConvolutionKernel kernel,
    _KLT_FloatImage imgout)
{
  float *indata  = imgin->data;
  float *outdata = imgout->data;
  int radius = kernel.width / 2;
  int ncols  = imgin->ncols;
  int nrows  = imgin->nrows;
  int i, j, k;

  /* Kernel width must be odd */
  assert(kernel.width % 2 == 1);

  /* Must read from and write to different images */
  assert(imgin != imgout);

  /* Output image must be large enough to hold result */
  assert(imgout->ncols >= imgin->ncols);
  assert(imgout->nrows >= imgin->nrows);

  /* Parallelize over all pixels (j,i) */
  #pragma acc parallel loop collapse(2) present(indata[0:ncols*nrows], outdata[0:ncols*nrows], kernel)
  for (j = 0; j < nrows; j++) {
    for (i = 0; i < ncols; i++) {
      float sum = 0.0f;
      if (i < radius || i >= ncols - radius) {
        /* Zero borders */
        outdata[j * ncols + i] = 0.0f;
      } else {
        /* Convolve middle columns with kernel */
        int base = j * ncols + (i - radius);
        for (k = 0; k < kernel.width; k++) {
          sum += indata[base + k] * kernel.data[kernel.width - 1 - k];
        }
        outdata[j * ncols + i] = sum;
      }
    }
  }
}

 
 /*********************************************************************
  * _convolveImageVert
  */
 
  static void _convolveImageVert(
    _KLT_FloatImage imgin,
    ConvolutionKernel kernel,
    _KLT_FloatImage imgout)
{
  float *indata  = imgin->data;
  float *outdata = imgout->data;
  int radius = kernel.width / 2;
  int ncols  = imgin->ncols;
  int nrows  = imgin->nrows;
  int i, j, k;

  /* Kernel width must be odd */
  assert(kernel.width % 2 == 1);

  /* Must read from and write to different images */
  assert(imgin != imgout);

  /* Output image must be large enough to hold result */
  assert(imgout->ncols >= imgin->ncols);
  assert(imgout->nrows >= imgin->nrows);

  /* Parallelize over all pixels (j,i) */
  #pragma acc parallel loop collapse(2) present(indata[0:ncols*nrows], outdata[0:ncols*nrows], kernel)
  for (j = 0; j < nrows; j++) {
    for (i = 0; i < ncols; i++) {
      float sum = 0.0f;
      if (j < radius || j >= nrows - radius) {
        /* Zero borders */
        outdata[j * ncols + i] = 0.0f;
      } else {
        /* Convolve middle rows with kernel */
        int base = (j - radius) * ncols + i;
        for (k = 0; k < kernel.width; k++) {
          sum += indata[base + k * ncols] * kernel.data[kernel.width - 1 - k];
        }
        outdata[j * ncols + i] = sum;
      }
    }
  }
}

 
/*********************************************************************
 * _convolveSeparate
 */

/* GPU-accelerated separable convolution with nested data region support */
static void _convolveSeparate(
 _KLT_FloatImage imgin,
 ConvolutionKernel horiz_kernel,
 ConvolutionKernel vert_kernel,
 _KLT_FloatImage imgout)
{
int ncols = imgin->ncols;
int nrows = imgin->nrows;
int npix  = ncols * nrows;

/* Temporary image */
_KLT_FloatImage tmpimg;

/* Allocate temporary image (host-side pointer) */
tmpimg = _KLTCreateFloatImage(ncols, nrows);

/* GPU data region with present_or_* for nested region support
 * - present_or_copyin: Use present if data already on GPU, else copyin
 * - present_or_copyout: Use present if data already on GPU, else copyout
 * - create: Temporary image only exists on GPU (no transfer)
 * This allows function to work both standalone AND when called from parent data region
 */
#pragma acc data present_or_copyin(imgin->data[0:npix], \
                                     horiz_kernel.data[0:MAX_KERNEL_WIDTH], \
                                     vert_kernel.data[0:MAX_KERNEL_WIDTH]) \
                 create(tmpimg->data[0:npix]) \
                 present_or_copyout(imgout->data[0:npix])
{
  _convolveImageHoriz(imgin, horiz_kernel, tmpimg);
  _convolveImageVert(tmpimg, vert_kernel, imgout);
}

/* Free memory */
_KLTFreeFloatImage(tmpimg);
} 
 
 /*********************************************************************
  * _KLTSetGlobalTrackingContext
  * 
  * Sets the global tracking context for persistent GPU buffer access.
  * This must be called before any pyramid/convolution operations.
  */
 
 
/*********************************************************************
 * _KLTComputeGradients
 */

void _KLTComputeGradients(
    _KLT_FloatImage img,
    float sigma,
    _KLT_FloatImage gradx,
    _KLT_FloatImage grady)
{

  /* Output images must be large enough to hold result */
  assert(gradx->ncols >= img->ncols);
  assert(gradx->nrows >= img->nrows);
  assert(grady->ncols >= img->ncols);
  assert(grady->nrows >= img->nrows);

  /* Compute kernels, if necessary */
  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

  int ncols = img->ncols;
  int nrows = img->nrows;
  int npix = ncols * nrows;

  /* Wrap BOTH convolution calls in single data region
   * - Input image transferred once
   * - Both gradients computed on GPU
   * - Results transferred back once
   * This eliminates redundant transfers between the two convolutions
   */
  #pragma acc data copyin(img->data[0:npix]) \
                   copyout(gradx->data[0:npix], grady->data[0:npix])
  {
    _convolveSeparate(img, gaussderiv_kernel, gauss_kernel, gradx);
    _convolveSeparate(img, gauss_kernel, gaussderiv_kernel, grady);
  }
}
 
/*********************************************************************
 * _KLTComputeSmoothedImage
 */

void _KLTComputeSmoothedImage(
    _KLT_FloatImage img,
    float sigma,
    _KLT_FloatImage smooth)
{
  /* Output image must be large enough to hold result */
  assert(smooth->ncols >= img->ncols);
  assert(smooth->nrows >= img->nrows);

  /* Compute kernel, if necessary; gauss_deriv is not used */
  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

  int ncols = img->ncols;
  int nrows = img->nrows;
  int npix = ncols * nrows;

  /* Data region with present_or_* for nested region support
   * Allows this function to work both standalone and when called from pyramid
   */
  #pragma acc data present_or_copyin(img->data[0:npix]) \
                   present_or_copyout(smooth->data[0:npix])
  {
    _convolveSeparate(img, gauss_kernel, gauss_kernel, smooth);
  }
}