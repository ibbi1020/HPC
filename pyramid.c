/*********************************************************************
 * pyramid.c
 *
 *********************************************************************/

/* Standard includes */
#include <assert.h>
#include <stdlib.h>		/* malloc() ? */
#include <string.h>		/* memset() ? */
#include <math.h>		/* */

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolve.h"	/* for computing pyramid */
#include "pyramid.h"


/*********************************************************************
 *
 */

_KLT_Pyramid _KLTCreatePyramid(
  int ncols,
  int nrows,
  int subsampling,
  int nlevels)
{
  _KLT_Pyramid pyramid;
  int nbytes = sizeof(_KLT_PyramidRec) +	
    nlevels * sizeof(_KLT_FloatImage *) +
    nlevels * sizeof(int) +
    nlevels * sizeof(int);
  int i;

  if (subsampling != 2 && subsampling != 4 && 
      subsampling != 8 && subsampling != 16 && subsampling != 32)
    KLTError("(_KLTCreatePyramid)  Pyramid's subsampling must "
             "be either 2, 4, 8, 16, or 32");

     
  /* Allocate memory for structure and set parameters */
  pyramid = (_KLT_Pyramid)  malloc(nbytes);
  if (pyramid == NULL)
    KLTError("(_KLTCreatePyramid)  Out of memory");
     
  /* Set parameters */
  pyramid->subsampling = subsampling;
  pyramid->nLevels = nlevels;
  pyramid->img = (_KLT_FloatImage *) (pyramid + 1);
  pyramid->ncols = (int *) (pyramid->img + nlevels);
  pyramid->nrows = (int *) (pyramid->ncols + nlevels);

  /* Allocate memory for each level of pyramid and assign pointers */
  for (i = 0 ; i < nlevels ; i++)  {
    pyramid->img[i] =  _KLTCreateFloatImage(ncols, nrows);
    pyramid->ncols[i] = ncols;  pyramid->nrows[i] = nrows;
    ncols /= subsampling;  nrows /= subsampling;
  }

  return pyramid;
}


/*********************************************************************
 *
 */

void _KLTFreePyramid(
  _KLT_Pyramid pyramid)
{
  int i;

  /* Free images */
  for (i = 0 ; i < pyramid->nLevels ; i++)
    _KLTFreeFloatImage(pyramid->img[i]);

  /* Free structure */
  free(pyramid);
}


/*********************************************************************
 *
 */

void _KLTComputePyramid(
    _KLT_FloatImage img, 
    _KLT_Pyramid pyramid,
    float sigma_fact)
{
  _KLT_FloatImage currimg, tmpimg;
  int ncols = img->ncols, nrows = img->nrows;
  int subsampling = pyramid->subsampling;
  int subhalf = subsampling / 2;
  float sigma = subsampling * sigma_fact;  /* empirically determined */
  int oldncols, oldnrows;
  int i, x, y;
  int src_x, src_y;
  ConvolutionKernel gauss_kernel, gaussderiv_kernel;

  if (subsampling != 2 && subsampling != 4 && 
      subsampling != 8 && subsampling != 16 && subsampling != 32)
    KLTError("(_KLTComputePyramid)  Pyramid's subsampling must "
             "be either 2, 4, 8, 16, or 32");

  assert(pyramid->ncols[0] == img->ncols);
  assert(pyramid->nrows[0] == img->nrows);

  /* Ensure Gaussian kernel is computed for this sigma */
  int dummy_width;
  _KLTGetKernelWidths(sigma, &dummy_width, &dummy_width);
  
  /* Get copies of the Gaussian kernels */
  _KLTGetGaussianKernels(&gauss_kernel, &gaussderiv_kernel);

  /* Allocate a single temporary image at full resolution and reuse it */
  tmpimg = _KLTCreateFloatImage(ncols, nrows);

  /* === SINGLE DATA REGION FOR ENTIRE PYRAMID CONSTRUCTION ===
   * This outer region keeps ALL pyramid data on GPU during construction.
   * Kernel data is transferred once and reused for all levels.
   * This eliminates all intermediate CPU<->GPU transfers!
   */
  #pragma acc data \
      copyin(img->data[0:ncols*nrows], \
             gauss_kernel.data[0:MAX_KERNEL_WIDTH]) \
      create(tmpimg->data[0:ncols*nrows], \
             pyramid->img[0]->data[0:ncols*nrows])
  {
    /* Copy original image to level 0 (parallel on GPU) */
    #pragma acc parallel loop
    for (i = 0; i < ncols * nrows; i++) {
      pyramid->img[0]->data[i] = img->data[i];
    }

    /* Current image starts as level 0 of the pyramid */
    currimg = pyramid->img[0];

    /* Build remaining pyramid levels (all on GPU) */
    for (i = 1; i < pyramid->nLevels; i++) {

      oldncols = currimg->ncols;
      oldnrows = currimg->nrows;

      /* Compute size of next pyramid level */
      int new_ncols = oldncols / subsampling;
      int new_nrows = oldnrows / subsampling;

      /* Nested data region for this level
       * - present(): currimg and tmpimg already on GPU from outer region
       * - create(): allocate new pyramid level on GPU only
       * _KLTComputeSmoothedImage will use present_or_* internally,
       * detecting data is already present and avoiding transfers
       */
      #pragma acc data \
          present(currimg->data[0:oldncols*oldnrows]) \
          present(tmpimg->data[0:oldncols*oldnrows]) \
          create(pyramid->img[i]->data[0:new_ncols*new_nrows])
      {
        /* Smooth current image into tmpimg (stays on GPU) */
        _KLTComputeSmoothedImage(currimg, sigma, tmpimg);

        /* GPU-accelerated subsampling: tmpimg -> pyramid->img[i] */
        #pragma acc parallel loop collapse(2) \
            present(tmpimg->data[0:oldncols*oldnrows]) \
            present(pyramid->img[i]->data[0:new_ncols*new_nrows])
        for (y = 0; y < new_nrows; y++) {
          for (x = 0; x < new_ncols; x++) {

            src_y = subsampling * y + subhalf;
            src_x = subsampling * x + subhalf;

            pyramid->img[i]->data[y * new_ncols + x] =
              tmpimg->data[src_y * oldncols + src_x];
          }
        }
      }

      /* Reassign current image to the new level */
      currimg = pyramid->img[i];
    }
  }
  /* === END DATA REGION - Pyramid now complete on GPU === */

  _KLTFreeFloatImage(tmpimg);
}


 











