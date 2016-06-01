#include "sopt/config.h"
#include <fstream>
#include <tiff.h>
#include <tiffio.h>
#include "sopt/types.h"
#include "sopt/exception.h"
#include "sopt/logging.h"
#include "sopt/utilities.h"

namespace {
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
  uint32_t result = 0;
  uint8_t *ptr = (uint8_t *)&result;
  auto const g = [](double p) -> uint8_t {
    auto const scaled = 255e0 * p;
    if(scaled < 0)
      return 0;
    return scaled > 255 ? 255 : uint8_t(scaled);
  };
  ptr[0] = g(pixel);
  ptr[1] = g(pixel);
  ptr[2] = g(pixel);
  ptr[3] = 255;
  return result;
}
}

namespace sopt {
namespace utilities {
Image<> read_tiff(std::string const &filename) {
  SOPT_INFO("Reading image file {} ", filename);
  TIFF *tif = TIFFOpen(filename.c_str(), "r");
  if(not tif)
    SOPT_THROW("Could not open file ") << filename;

  uint32 width, height, t;

  TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
  TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
  TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &t);
  SOPT_DEBUG("- image size {}, {} ", width, height);
  Image<> result = Image<>::Zero(height, width);

  uint32 *raster = (uint32 *)_TIFFmalloc(width * height * sizeof(uint32));
  if(not raster)
    SOPT_THROW("Could not allocate memory to read file ") << filename;
  if(not TIFFReadRGBAImage(tif, width, height, raster, 0))
    SOPT_THROW("Could not read file ") << filename;

  uint32_t *pixel = (uint32_t *)raster;
  for(uint32 i(0); i < height; ++i)
    for(uint32 j(0); j < width; ++j, ++pixel)
      result(i, j) = convert_to_greyscale(*pixel);

  _TIFFfree(raster);

  TIFFClose(tif);
  return result;
}

void write_tiff(Image<> const &image, std::string const &filename) {
  SOPT_INFO("Writing image file {} ", filename);
  SOPT_DEBUG("- image size {}, {} ", image.rows(), image.cols());
  TIFF *tif = TIFFOpen(filename.c_str(), "w");
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
} // utilities
} // sopt
