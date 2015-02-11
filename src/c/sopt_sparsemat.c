/*! 
 * \file sopt_sparsemat.c
 * Functionality to perform operations with sparse matrices.
 */
#include "sopt_config.h"

#include <stdlib.h>
#include "sopt_sparsemat.h"
#include "sopt_error.h"


/*!
 * Free all memory used to store a spare matrix.
 *
 * \param[in] mat Sparse matrix to free.
 *
 * \authors <a href="http://www.jasonmcewen.org">Jason McEwen</a>
 */
void sopt_sparsemat_free(sopt_sparsemat *mat) {

  if(mat->vals != NULL) free(mat->vals);
  if(mat->cvals != NULL) free(mat->cvals);
  if(mat->rowind != NULL) free(mat->rowind);
  if(mat->colptr != NULL) free(mat->colptr);
  mat->nrows = 0;
  mat->ncols = 0;
  mat->nvals = 0;

}


/*!
 * Compute explicit representation of a sparse matrix.
 *
 * \param[out] A Explicit matrix representation of sparse matrix S.
 * The matrix A is stored in column-major order, i.e. can be accessed as
 * A[col_index * nrows + row_index].
 * \param[in] S Sparse matrix to representation explicitly (passed by
 * reference).
 *
 * \note Space for the output matrix A is allocated herein and must be
 * freed by the calling routine.
 *
 * \authors <a href="http://www.jasonmcewen.org">Jason McEwen</a>
 */
void sopt_sparsemat_explictmat(double **A, sopt_sparsemat *S) {

  int c, rr;

  // Allocate space for explicit matrix (initialised with zeros).
  *A = (double*)calloc(S->nrows * S->ncols, sizeof(double));
  SOPT_ERROR_MEM_ALLOC_CHECK(*A);

  // Construct explicit matrix.
  for (c = 0; c < S->ncols; c++)
    for (rr = S->colptr[c]; rr < S->colptr[c+1]; rr++)
      (*A)[c * S->nrows + S->rowind[rr]] = S->vals[rr];

}


/*!
 * Multiple a real vector by a real sparse matrix, i.e. compute \f$y =
 * A x\f$.
 *
 * \param[out] y Ouput vector of length nrows.
 * \param[in] x Input vector of length ncols.
 * \param[in] A Sparse matrix (passed by reference).
 *
 * \note Space for the output vector must be allocated by the calling
 * routine.
 *
 * \authors <a href="http://www.jasonmcewen.org">Jason McEwen</a>
 */
void sopt_sparsemat_fwd_real(double *y, double *x, sopt_sparsemat *A) {

  int r, rr, c;

  for (r = 0; r < A->nrows; r++)
    y[r] = 0.0;

  for (c = 0; c < A->ncols; c++)
    for (rr = A->colptr[c]; rr < A->colptr[c+1]; rr++)
      y[A->rowind[rr]] += A->vals[rr] * x[c];
 
}


/*!
 * Multiply a real vector by the adjoint of a real sparse matrix,
 * i.e. compute \f$y = A^H x\f$, where \f$H\f$ is the Hermitian operator.
 *
 * \param[out] y Ouput vector of length ncols.
 * \param[in] x Input vector of length nrows.
 * \param[in] A Sparse matrix (passed by reference).
 *
 * \note Space for the output vector must be allocated by the calling
 * routine.
 *
 * \authors <a href="http://www.jasonmcewen.org">Jason McEwen</a>
 */
void sopt_sparsemat_adj_real(double *y, double *x, sopt_sparsemat *A) {

  int rr, c;

  for (c = 0; c < A->ncols; c++) {
    y[c] = 0.0;
    for (rr = A->colptr[c]; rr < A->colptr[c+1]; rr++)
      y[c] += A->vals[rr] * x[A->rowind[rr]];
  }

}


/*!
 * Multiply a complex vector by a real sparse matrix, i.e. compute
 * \f$y = A x\f$.
 *
 * \param[out] y Ouput vector of length nrows.
 * \param[in] x Input vector of length ncols.
 * \param[in] A Sparse matrix (passed by reference).
 *
 * \note Space for the output vector must be allocated by the calling
 * routine.
 *
 * \authors <a href="http://www.jasonmcewen.org">Jason McEwen</a>
 */
void sopt_sparsemat_fwd_complex(complex double *y, complex double *x,
				  sopt_sparsemat *A) {

  int r, rr, c;

  for (r = 0; r < A->nrows; r++)
    y[r] = 0.0 + 0.0*I;

  for (c = 0; c < A->ncols; c++)
    for (rr = A->colptr[c]; rr < A->colptr[c+1]; rr++)
      y[A->rowind[rr]] += A->cvals[rr] * x[c];
 
}


/*!
 * Multiple a complex vector by the adjoint of a real sparse matrix,
 * i.e. compute \f$y = A^H x\f$, where \f$H\f$ is the Hermitian operator.
 *
 * \param[out] y Ouput vector of length ncols.
 * \param[in] x Input vector of length nrows.
 * \param[in] A Sparse matrix (passed by reference).
 *
 * \note Space for the output vector must be allocated by the calling
 * routine.
 *
 * \authors <a href="http://www.jasonmcewen.org">Jason McEwen</a>
 */
void sopt_sparsemat_adj_complex(complex double *y, complex double *x,
				  sopt_sparsemat *A) {

  int rr, c;

  for (c = 0; c < A->ncols; c++) {
    y[c] = 0.0 + 0.0*I;
    for (rr = A->colptr[c]; rr < A->colptr[c+1]; rr++)
      y[c] += conj(A->cvals[rr]) * x[A->rowind[rr]];
  }

}
