/*! \file sopt_error.c
 *  Error functions used in SOPT package.
 */

#include "sopt_error.h"

/*!
 * Display error message and halt program execution.
 *
 * \param[in] comment Additional comment to display.
 *
 * \authors <a href="http://www.jasonmcewen.org">Jason McEwen</a>
 */
extern inline void SOPT_ERROR_GENERIC(char *comment);

/*!
 * Test whether memory allocation was successful (i.e. check allocated
 * pointer not NULL) and throw a generic error if allocation failed.
 *
 * \param[in] pointer Pointer that should point to allocated memory is
 * memory allocation was successful.
 *
 * \authors <a href="http://www.jasonmcewen.org">Jason McEwen</a>
 */
extern inline void SOPT_ERROR_MEM_ALLOC_CHECK(void *pointer);
