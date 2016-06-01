#ifndef SOPT_RAN
#define SOPT_RAN
#include "sopt_config.h"

#ifdef __cplusplus
extern "C" {
#endif

double sopt_ran_gasdev2(int idum);
double sopt_ran_ran2(int idum);
int sopt_ran_knuthshuffle(int *perm, int nperm, int N, int seed);


#ifdef __cplusplus
}
#endif
#endif
