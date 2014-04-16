
/*! 
 * \file sopt_about.c
 * Program to display summary information, including version and build numbers.
 */
#include <sopt_config.h>

#include <stdio.h>

int main(int argc, char *argv[]) {

  printf("%s\n", "==========================================================");
  printf("%s\n", "  SOPT: Sparse OPTimisation");
  printf("%s\n", "  By R. E. Carrillo, J. D. McEwen & Y. Wiaux");

  printf("%s\n", "  See README.txt for more information.");
  printf("%s\n", "  See LICENSE.txt for license information.");

  printf("%s%s\n", "  Version: ", sopt_version);
  printf("%s%s\n", "  Build:   ", sopt_gitref);
  printf("%s\n", "==========================================================");

  return 0;

}
