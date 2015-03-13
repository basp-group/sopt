/*! 
 * \file sopt_ran.c
 * Functionality to generate random deviates.
 */
#include "sopt_config.h"

#include <math.h>
#include <stdlib.h>
#include "sopt_ran.h"
#include "sopt_error.h"


/*!
 * Generate sample from Gaussian distribution of mean 0 and standard
 * deviation 1 given seed.  (Using double precision.)
 *
 * \param[in] idum Seed.
 * \retval ran_dp Generated Gaussian deviate.
 *
 * \todo [JDM] Test generation of Gaussian deviate (then remove warning
 * comment).
 *
 * \note Gaussian deviate inpsired from gasdev (Num Rec 1992, chap
 * 7.3), the only difference is the use of RAN2 instead of RAN1.
 * \note This routine calls \ref sopt_ran_ran2 and so changes the
 * state of the random generator.
 *
 * \warning This C implementation has been copied from a Fortran
 * version and has not yet been tested!
 *
 * \warning Must inialise \ref sopt_ran_ran2 first with a negative
 * seed first and then subsequently call with the identical magnitude
 * positive seed, i.e. if seed>0, initialise \ref sopt_ran_ran2 with
 * -seed and then call with seed.
 *
 * \author Numerical recipes
 */
double sopt_ran_gasdev2(int idum) {

  double gasdev2_dp;
  double fac,rsq,v1,v2;
  static int iset = 0;
  static double gset = 0.0;
    
  if (iset ==0) {
    do {
      v1=2.*sopt_ran_ran2(idum)-1.;
      v2=2.*sopt_ran_ran2(idum)-1.;
      rsq=v1*v1+v2*v2;
    } while(rsq >= 1.0 || rsq == 0.0);
    fac=sqrt(-2.*log(rsq)/rsq);
    gset=v1*fac;
    gasdev2_dp=v2*fac;
    iset=1;
  }
  else {
    gasdev2_dp=gset;
    iset=0;
  }
  
  return gasdev2_dp;
      
}


/*!  
 * Generate uniform deviate in range [0,1) given seed. (Using double
 * precision.)
 *
 * \param[in] idum Seed.
 * \retval ran_dp Generated uniform deviate.
 *
 * \note Uniform deviate (Num rec 1992, chap 7.1), original routine
 *  said to be 'perfect'.
 *
 * \note Long period (> 2 × 1018) random number generator of L’Ecuyer
 *  with Bays-Durham shuffle and added safeguards. Returns a uniform
 *  random deviate between 0.0 and 1.0 (exclusive of the endpoint
 *  values). Call with idum a negative integer to initialize;
 *  thereafter, do not alter idum between successive deviates in a
 *  sequence. RNMX should approximate the largest floating value that
 *  is less than 1.
 *
 * \warning Must inialise with a negative seed and then subsequently
 * call with the identical magnitude positive seed, i.e. if seed>0,
 * initialise with -seed and then call with seed.
 *
 * \author Numerical recipes
 */
double sopt_ran_ran2(int idum) {

  int IM1=2147483563,IM2=2147483399,IMM1=IM1-1, 
    IA1=40014,IA2=40692,IQ1=53668,IQ2=52774,IR1=12211,IR2=3791, 
    NTAB=32,NDIV=1+IMM1/NTAB;

  double AM=1./IM1,EPS=1.2e-7,RNMX=1.-EPS;
  int j,k;
  static int iv[32],iy,idum2 = 123456789; 
  // N.B. in C static variables are initialised to 0 by default.

  if (idum <= 0) {
    idum= (-idum>1 ? -idum : 1); // max(-idum,1);
    idum2=idum;
    for(j=NTAB+8;j>=1;j--) {
      k=idum/IQ1;
      idum=IA1*(idum-k*IQ1)-k*IR1;
      if (idum < 0) idum=idum+IM1;
      if (j < NTAB) iv[j-1]=idum;
    }
    iy=iv[0];
  }
  k=idum/IQ1;
  idum=IA1*(idum-k*IQ1)-k*IR1;
  if (idum < 0) idum=idum+IM1;
  k=idum2/IQ2;
  idum2=IA2*(idum2-k*IQ2)-k*IR2;
  if (idum2 < 0) idum2=idum2+IM2;
  j=1+iy/NDIV;
  iy=iv[j-1]-idum2;
  iv[j-1]=idum;
  if(iy < 1)iy=iy+IMM1;
  return (AM*iy < RNMX ? AM*iy : RNMX); // min(AM*iy,RNMX);

}


/*!  
 * Knuth shuffle to generate random permutation of integers.
 *
 * \param[out] perm Array of length nperm of random permutations of
 * integers from 0, N-1 (space must be allocated by calling routine).
 * \param[in] nperm Number of permutated integers to generate.
 * \param[in] N Number of integers to consider when generating the
 * random permutation.
 * \param[in] seed Seed for random number generator.
 * \retval error Zero indicates no errors occured.
 *
 * \note This routine calls \ref sopt_ran_ran2 and so changes the
 * state of the random generator.
 *
 * \authors <a href="http://www.jasonmcewen.org">Jason McEwen</a>
 */
int sopt_ran_knuthshuffle(int *perm, int nperm, int N, int seed) {

  int *scratch;
  int i, j, tmp, nmax;

  // Check inputs.
  if (nperm > N) {
    SOPT_ERROR_GENERIC("Number of required permutations greater than number of integers");
    return 1;
  }
      
  // Allocate scratch space.
  scratch = (int*)malloc(N * sizeof(int));
  SOPT_ERROR_MEM_ALLOC_CHECK(scratch);

  // Initialise scratch space.
  for (i = 0; i < N; i++)
    scratch[i] = i;

  // Initialise random number generator with negative seed.
  seed = abs(seed);
  sopt_ran_ran2(-seed);

  // Compute permutation.
  // If require nperm = N permutations then don't permute last element.
  nmax = nperm < N-1 ? nperm : N-1; // min(nperm, N-1)
  for (i = 0; i < nmax; i++) {

    // Generate random index j in [i,N-1].
    j = (int)(sopt_ran_ran2(seed) * (N - i));
    j = j + i;

    // Although j in [i,N-1] enforce this to be sure.
    j = (j >= N ? N-1 : j);
    j = (j < i ? i : j);

    // Swap entries i and j.
    tmp = scratch[i];
    scratch[i] = scratch[j];
    scratch[j] = tmp;

  }
    
  // Copy required data from scratch space.
  for (i = 0; i < nperm; i++)
    perm[i] = scratch[i];

  // Free memory.
  free(scratch);

  return 0;
}
