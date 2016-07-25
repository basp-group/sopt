#ifndef SOPT_LOGGING_H
#define SOPT_LOGGING_H

#include "sopt/config.h"

#ifdef SOPT_DO_LOGGING
#include "sopt/logging.enabled.h"
#else
#include "sopt/logging.disabled.h"
#endif

//! \macro Normal but significant condition or critical error
#define SOPT_NOTICE(...) SOPT_LOG_(, critical, __VA_ARGS__)
//! \macro Something is definitely wrong, algorithm exits
#define SOPT_ERROR(...) SOPT_LOG_(, error, __VA_ARGS__)
//! \macro Something might be going wrong
#define SOPT_WARN(...) SOPT_LOG_(, warn, __VA_ARGS__)
//! \macro Verbose informational message about normal condition
#define SOPT_INFO(...) SOPT_LOG_(, info, __VA_ARGS__)
//! \macro Output some debugging
#define SOPT_DEBUG(...) SOPT_LOG_(, debug, __VA_ARGS__)
//! \macro Output internal values of no interest to anyone
//! \details Except maybe when debugging.
#define SOPT_TRACE(...) SOPT_LOG_(, trace, __VA_ARGS__)

#endif
