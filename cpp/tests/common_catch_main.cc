#define CATCH_CONFIG_RUNNER

#include "catch.hpp"
#include "sopt/logging.h"

int main( int argc, char* const argv[] )
{
  Catch::Session session; // There must be exactly once instance

  int returnCode = session.applyCommandLine( argc, argv );
  if( returnCode != 0 ) // Indicates a command line error
    return returnCode;

# ifdef SOPT_DO_LOGGING
#  ifndef SOPT_TEST_DEBUG_LEVEL
#    define SOPT_TEST_DEBUG_LEVEL debug
#  endif
    auto const logger = sopt::logging::initialize();
    logger->set_level(spdlog::level::SOPT_TEST_DEBUG_LEVEL);
# endif

  return session.run();
}
