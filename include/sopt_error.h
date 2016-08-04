/*! \file sopt_error.h
 *  Error functions used in SOPT package.
 */

#ifndef SOPT_ERROR_H
#define SOPT_ERROR_H
#include "sopt_config.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

extern inline void SOPT_ERROR_GENERIC(char *comment);
extern inline void SOPT_ERROR_MEM_ALLOC_CHECK(void *pointer);

#ifdef __cplusplus
}
#endif
#endif
