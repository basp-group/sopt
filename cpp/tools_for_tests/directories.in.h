#ifndef SOPT_DATA_DIR_H
#define SOPT_DATA_DIR_H

#include <string>
namespace sopt { namespace notinstalled {

//! Holds images and such
inline std::string data_directory() { return "@PROJECT_SOURCE_DIR@/images"; }
//! Output artefacts from tests
inline std::string output_directory() { return "@PROJECT_BINARY_DIR@/outputs"; }

}} /* sopt::notinstalled */
#endif
