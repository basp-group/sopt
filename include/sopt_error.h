/*! \file sopt_error.h
 *  Error functions used in SOPT package.
 */

#ifndef SOPT_ERROR
#define SOPT_ERROR
#include "sopt_config.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

inline void SOPT_ERROR_GENERIC(char *comment) {
  printf("ERROR: %s.\n", comment);					
  printf("ERROR: %s <%s> %s %s %s %d.\n",				
    "Occurred in function",					
    __PRETTY_FUNCTION__,						
    "of file", __FILE__,						
    "on line", __LINE__);					
  exit(1);
}

inline void SOPT_ERROR_MEM_ALLOC_CHECK(void *pointer) {				
  if(pointer == NULL)
    SOPT_ERROR_GENERIC("Memory allocation failed");
}

#ifdef __cplusplus
}
#endif
#endif
