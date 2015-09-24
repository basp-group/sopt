#include <tiff.h>
#include <tiffio.h>
#include <fstream>

#include <sopt/exception.h>
#include "tiffwrappers.h"
#include "sopt/logging.h"

//! A single pixel
//! Converts ABGR to greyscale double value
double convert_to_greyscale(uint32_t &pixel) {
  uint8_t const blue = TIFFGetB(pixel);
  uint8_t const green = TIFFGetG(pixel);
  uint8_t const red = TIFFGetR(pixel);
  return static_cast<double>(blue + green + red) / (3e0 * 255e0);
}
//! Converts greyscale double value to RGBA
uint32_t convert_from_greyscale(double pixel) {
  assert(pixel >= 0e0 and pixel <= 10e0);
  uint8_t const value = static_cast<uint8_t>(std::round(pixel * 255e0));
  uint32_t result = 0;
  uint8_t *ptr = (uint8_t*)&result;
  ptr[0] = value;
  ptr[1] = value;
  ptr[2] = value;
  ptr[3] = 255;
  return result;
}



namespace sopt {
sopt::t_rMatrix read_tiff(std::string const & filename) {
  SOPT_INFO("Reading image file {} ", filename);
  TIFF* tif = TIFFOpen(filename.c_str(), "r");
  if(not tif)
    SOPT_THROW("Could not open file ") << filename;

  uint32 width, height, t;

  TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
  TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
  TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &t);
  SOPT_DEBUG("- image size {}, {} ", width, height);
  sopt::t_rMatrix result = sopt::t_rMatrix::Zero(height, width);

  uint32* raster = (uint32*) _TIFFmalloc(width * height * sizeof (uint32));
  if(not raster)
    SOPT_THROW("Could not allocate memory to read file ") << filename;
  if (not TIFFReadRGBAImage(tif, width, height, raster, 0))
    SOPT_THROW("Could not read file ") << filename;


  uint32_t *pixel = (uint32_t*)raster;
  for(uint32 i(0); i < height; ++i)
    for(uint32 j(0); j < width; ++j, ++pixel)
      result(i, j) = convert_to_greyscale(*pixel);

  _TIFFfree(raster);

  TIFFClose(tif);
  return result;
}

sopt::t_rMatrix read_standard_tiff(std::string const &name) {
  std::string const stdname = sopt::data_directory() + "/" + name + ".tiff";
  bool const is_std = std::ifstream(stdname).good();
  return sopt::read_tiff(is_std ? stdname: name);
}

void write_tiff(sopt::t_rMatrix const & image, std::string const & filename) {
  SOPT_INFO("Writing image file {} ", filename);
  SOPT_DEBUG("- image size {}, {} ", image.rows(), image.cols());
  TIFF* tif = TIFFOpen(filename.c_str(), "w");
  if(not tif)
    SOPT_THROW("Could not open file ") << filename;

  uint32 const width = image.cols();
  uint32 const height = image.rows();

  SOPT_TRACE("Allocating buffer");
  std::vector<uint32_t> raster(width * height);

  TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
  TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
  TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, sizeof(decltype(raster)::value_type));
  TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8);
  TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, height);
  TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
  TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
  TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_BOTLEFT);

  SOPT_TRACE("Initializing buffer");
  auto pixel = raster.begin();
  for(uint32 i(0); i < height; ++i)
    for(uint32 j(0); j < width; ++j, ++pixel)
      *pixel = convert_from_greyscale(image(i, j));

  SOPT_TRACE("Writing strip");
  TIFFWriteEncodedStrip(tif, 0, &raster[0], width * height * sizeof(decltype(raster)::value_type));

  TIFFWriteDirectory(tif);
  SOPT_TRACE("Closing tif");
  TIFFClose(tif);
  SOPT_TRACE("Freeing raster");
}
} /* sopt  */
