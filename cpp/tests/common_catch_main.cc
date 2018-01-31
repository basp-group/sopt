#define CATCH_CONFIG_RUNNER

#include "sopt/config.h"
#include <catch.hpp>
#include <memory>
#include <random>
#include "sopt/logging.h"

std::unique_ptr<std::mt19937_64> mersenne(new std::mt19937_64(0));

int main(int argc, char **argv) {
  Catch::Session session; // There must be exactly once instance

  int returnCode = session.applyCommandLine(argc, argv);
  if(returnCode != 0) // Indicates a command line error
    return returnCode;
  mersenne.reset(new std::mt19937_64(session.configData().rngSeed));

  sopt::logging::initialize();

  return session.run();
}
