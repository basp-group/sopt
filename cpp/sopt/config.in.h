#ifndef SOPT_CPP_CONFIG_H

//! Problems with using and constructors
#cmakedefine SOPT_HAS_USING
#ifndef SOPT_HAS_USING
#define SOPT_HAS_NOT_USING
#endif

//! Macro to start logging or not
#cmakedefine SOPT_DO_LOGGING
#ifndef SOPT_TEST_DEBUG_LEVEL
#define SOPT_TEST_DEBUG_LEVEL "debug"
#endif

#endif
