/*! 
 * \file sopt_image.c
 * Functionality to read and write image files.
 */
#include "sopt_config.h"

#include <stdlib.h>
#include <tiffio.h>
#include "sopt_error.h"


/*!
 * Read RGBA tiff image from file as greyscale image (the red channel
 * of the RGBA image is read).
 * 
 * \note Memory for img is allocated herein and must be freed by the
 * calling routine.
 *
 * \param[out] img Greyscale image read from tiff file.
 * \param[out] nx Dimension of image in x direction.
 * \param[out] ny Dimension of image in y direction.
 * \param[in] filename Name of file to read tiff image from.
 * \param[in] scale Scale factor to apply to image pixels (8 bit
 * values read from tiff file are mapped to (val-offset)*scale.
 * \param[in] offset Offset to apply to image pixels (8 bit
 * values read from tiff file are mapped to (val-offset)*scale.
 */
int sopt_image_tiff_read(double **img, int *nx, int *ny, char *filename, 
			 double scale, double offset) {

  size_t npixels;
  uint32 *raster;
  //uint16* orientation;
  int i, j;
  TIFF* tif;
  int fail = 0;

  // Open file and get fields.
  tif = TIFFOpen(filename, "r");
  SOPT_ERROR_MEM_ALLOC_CHECK(tif);
  TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, nx);
  TIFFGetField(tif, TIFFTAG_IMAGELENGTH, ny);
  //TIFFGetField(tif, TIFFTAG_ORIENTATION, &orientation);

  // Allocate space for image.
  npixels = (*nx) * (*ny);
  raster = (uint32*) _TIFFmalloc(npixels * sizeof (uint32));
  SOPT_ERROR_MEM_ALLOC_CHECK(raster);
  *img = (double*)malloc(npixels * sizeof(double));
  SOPT_ERROR_MEM_ALLOC_CHECK(img);

  // Read image.
  if (TIFFReadRGBAImage(tif, *nx, *ny, raster, 0)) {    
    for (j=0; j<*ny; j++) {
      for (i=0; i<*nx; i++) {
	(*img)[j*(*nx) + i] = (TIFFGetR(raster[j*(*nx) + i]) - offset) * scale;
      }
    }
    fail = 0;
  }
  else {
    fail = 1;
  }

  // Free temporary memory and close file.
  _TIFFfree(raster);
  TIFFClose(tif);

  return fail;
  
}


/*!
 * Write RGBA tiff image to file from greyscale image (the greyscale
 * image is written to the red, green and blue channels of the RGBA
 * image).
 * 
 * \param[in] img Greyscale image written to tiff file.
 * \param[in] nx Dimension of image in x direction.
 * \param[in] ny Dimension of image in y direction.
 * \param[in] filename Name of file to write tiff image to.
 * \param[in] scale Scale factor to apply to image pixels (8 bit
 * values written to tiff file are transformed by (img-offset)*scale.
 * \param[in] offset Offset to apply to image pixels (8 bit
 * values written to tiff file are transformed by (img-offset)*scale.
 */
int sopt_image_tiff_write(double *img, int nx, int ny, char *filename, 
			  double scale, double offset) {

  size_t npixels;
  uint32 *raster;
  int i, j;
  TIFF* tif;
  int fail = 0;
  uint8 pix;
  
  // Construct raster data to write to file.
  npixels = nx * ny;
  raster = (uint32*) _TIFFmalloc(npixels * sizeof (uint32));
  SOPT_ERROR_MEM_ALLOC_CHECK(raster);
  for (j=0; j<ny; j++) {
      for (i=0; i<nx; i++) {
	pix = (uint8)((img[j*nx + i] - offset) * scale);
	raster[j*nx + i] = 255 << 24 | pix << 16 | pix << 8 | pix; 
      }
    }

  // Open file and set fields.
  tif = TIFFOpen(filename, "w");
  TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, nx);
  TIFFSetField(tif, TIFFTAG_IMAGELENGTH, ny);
  TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8);
  TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 4);
  TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_LZW); // LZW is lossless
  TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
  TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_BOTLEFT); // alternative TOPLEFT

  // Write file.
  if(TIFFWriteEncodedStrip(tif, 0, raster, npixels*4))
    fail = 0;
  else
    fail = 1;

  // Free temporary memory and close file.
  _TIFFfree(raster);
  TIFFClose(tif);
  
  return fail;

}
