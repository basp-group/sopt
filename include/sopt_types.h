/*! \file sopt_types.h
 *  Types and constants used in SOPT package.
 */

#ifndef SOPT_TYPES
#define SOPT_TYPES
#include "sopt_config.h"

#ifdef __cplusplus
#include <complex>
#else
#include <complex.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SOPT_STRLEN 256

#define SOPT_PI    3.141592653589793238462643383279502884197
#define SOPT_PION2 1.570796326794896619231321691639751442099

#define SOPT_SQRT2 1.41421356237309504880168872420969807856967

#ifdef __cplusplus
typedef std::complex<double> sopt_complex_double;
#else
typedef complex double sopt_complex_double;
#endif


#ifdef __cplusplus
}
#endif
#endif
