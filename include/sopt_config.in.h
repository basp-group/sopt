#ifndef SOPT_CONFIG_IN_H
#define SOPT_CONFIG_IN_H

// Deal with the different ways to include blas
#cmakedefine SOPT_BLAS_H <@SOPT_BLAS_H@>

//! Library version
static const char sopt_version[] = "@Sopt_VERSION@";
//! Hash of the current git reference
static const char sopt_gitref[] = "@Sopt_GITREF@";
#endif
