#define CATCH_CONFIG_RUNNER

#include <catch.hpp>
#include "sopt/logging.h"

int main( int argc, char* const argv[] )
{
  Catch::Session session; // There must be exactly once instance

  int returnCode = session.applyCommandLine( argc, argv );
  if( returnCode != 0 ) // Indicates a command line error
    return returnCode;

  sopt::logging::initialize();
  sopt::logging::set_level(SOPT_TEST_DEBUG_LEVEL);

  return session.run();
}
