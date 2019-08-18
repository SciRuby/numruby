/*
 * GEQRF computes a QR factorization of a real M-by-N matrix A:
 * A = Q * R.
 *
 */
VALUE nm_geqrf(int argc, VALUE* argv) {
  nmatrix* matrix;
  Data_Get_Struct(argv[0], nmatrix, matrix);

  int m = matrix->shape[0]; //no. of rows
  int n = matrix->shape[1]; //no. of cols
  int lda = n, info = -1;

  nmatrix* result_qr = nmatrix_new(matrix->dtype, matrix->stype, 2, matrix->count, matrix->shape, NULL);
  nmatrix* result_tau = nmatrix_new(matrix->dtype, matrix->stype, 1, min(m, n), NULL, NULL);
  result_tau->shape[0] = min(m, n);

  switch(matrix->dtype) {
    case nm_bool:
    {
      //not supported error
      break;
    }
    case nm_int:
    {
      //not supported error
      break;
    }
    case nm_float32:
    {
      float* tau_elements = ALLOC_N(float, min(m, n));
      float* elements = ALLOC_N(float, matrix->count);
      memcpy(elements, matrix->elements, sizeof(float)*matrix->count);
      info = LAPACKE_sgeqrf(LAPACK_ROW_MAJOR, m, n, elements, lda, tau_elements);

      result_qr->elements = elements;
      result_tau->elements = tau_elements;

      VALUE qr = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_qr);
      VALUE tau = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_tau);
      return rb_ary_new3(2, qr, tau);
      break;
    }
    case nm_float64:
    {
      double* tau_elements = ALLOC_N(double, min(m, n));
      double* elements = ALLOC_N(double, matrix->count);
      memcpy(elements, matrix->elements, sizeof(double)*matrix->count);
      info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, elements, lda, tau_elements);

      result_qr->elements = elements;
      result_tau->elements = tau_elements;

      VALUE qr = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_qr);
      VALUE tau = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_tau);
      return rb_ary_new3(2, qr, tau);
      break;
    }
    case nm_complex32:
    {
      float complex* tau_elements = ALLOC_N(float complex, min(m, n));
      float complex* elements = ALLOC_N(float complex, matrix->count);
      memcpy(elements, matrix->elements, sizeof(float complex)*matrix->count);
      info = LAPACKE_cgeqrf(LAPACK_ROW_MAJOR, m, n, elements, lda, tau_elements);

      result_qr->elements = elements;
      result_tau->elements = tau_elements;

      VALUE qr = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_qr);
      VALUE tau = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_tau);
      return rb_ary_new3(2, qr, tau);
      break;
    }
    case nm_complex64:
    {
      double complex* tau_elements = ALLOC_N(double complex, min(m, n));
      double complex* elements = ALLOC_N(double complex, matrix->count);
      memcpy(elements, matrix->elements, sizeof(double complex)*matrix->count);
      info = LAPACKE_zgeqrf(LAPACK_ROW_MAJOR, m, n, elements, lda, tau_elements);

      result_qr->elements = elements;
      result_tau->elements = tau_elements;

      VALUE qr = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_qr);
      VALUE tau = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_tau);
      return rb_ary_new3(2, qr, tau);
      break;
    }
  }
  return INT2NUM(-1);
}

/*
 * ORGQR generates an M-by-N real matrix Q with orthonormal columns,
 * which is defined as the first N columns of a product of K elementary
 * reflectors of order M
 *       Q = H(1) H(2) . . . H(k)
 * as returned by GEQRF.
 */
VALUE nm_orgqr(int argc, VALUE* argv) {
  nmatrix* matrix_qr;
  Data_Get_Struct(argv[0], nmatrix, matrix_qr);

  nmatrix* matrix_tau;
  Data_Get_Struct(argv[1], nmatrix, matrix_tau);

  int m = matrix_qr->shape[0]; //no. of rows
  int n = matrix_qr->shape[1]; //no. of cols
  int k = matrix_tau->shape[0];
  int lda = n, info = -1;

  nmatrix* result_q = nmatrix_new(matrix_qr->dtype, matrix_qr->stype, 2, matrix_qr->count, matrix_qr->shape, NULL);

  switch(matrix_qr->dtype) {
    case nm_bool:
    {
      //not supported error
      break;
    }
    case nm_int:
    {
      //not supported error
      break;
    }
    case nm_float32:
    {
      float* tau_elements = (float*)matrix_tau->elements;
      float* elements = ALLOC_N(float, matrix_qr->count);
      memcpy(elements, matrix_qr->elements, sizeof(float)*matrix_qr->count);
      info = LAPACKE_sorgqr(LAPACK_ROW_MAJOR, m, n, k, elements, lda, tau_elements);

      result_q->elements = elements;

      return Data_Wrap_Struct(NMatrix, NULL, nm_free, result_q);
      break;
    }
    case nm_float64:
    {
      double* tau_elements = (double*)matrix_tau->elements;
      double* elements = ALLOC_N(double, matrix_qr->count);
      memcpy(elements, matrix_qr->elements, sizeof(double)*matrix_qr->count);
      info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, m, n, k, elements, lda, tau_elements);

      result_q->elements = elements;

      return Data_Wrap_Struct(NMatrix, NULL, nm_free, result_q);
      break;
    }
    case nm_complex32:
    {
      float complex* tau_elements = (float complex*)matrix_tau->elements;
      float complex* elements = ALLOC_N(float complex, matrix_qr->count);
      memcpy(elements, matrix_qr->elements, sizeof(float complex)*matrix_qr->count);
      info = LAPACKE_cungqr(LAPACK_ROW_MAJOR, m, n, k, elements, lda, tau_elements);

      result_q->elements = elements;

      return Data_Wrap_Struct(NMatrix, NULL, nm_free, result_q);
      break;
    }
    case nm_complex64:
    {
      double complex* tau_elements = (double complex*)matrix_tau->elements;
      double complex* elements = ALLOC_N(double complex, matrix_qr->count);
      memcpy(elements, matrix_qr->elements, sizeof(double complex)*matrix_qr->count);
      info = LAPACKE_zungqr(LAPACK_ROW_MAJOR, m, n, k, elements, lda, tau_elements);

      result_q->elements = elements;

      return Data_Wrap_Struct(NMatrix, NULL, nm_free, result_q);
      break;
    }
  }
  return INT2NUM(-1);
}

/*
 * GEQP3 computes a QR factorization with column pivoting of a
 * matrix A: A*P = Q*R using Level 3 BLAS.
 *
 */
VALUE nm_geqp3(int argc, VALUE* argv) {
  nmatrix* matrix;
  Data_Get_Struct(argv[0], nmatrix, matrix);

  int m = matrix->shape[0]; //no. of rows
  int n = matrix->shape[1]; //no. of cols
  int lda = n, info = -1;

  nmatrix* result_qr = nmatrix_new(matrix->dtype, matrix->stype, 2, matrix->count, matrix->shape, NULL);
  nmatrix* result_tau = nmatrix_new(matrix->dtype, matrix->stype, 1, min(m, n), NULL, NULL);
  result_tau->shape[0] = min(m, n);
  nmatrix* result_jpvt = nmatrix_new(matrix->dtype, matrix->stype, 1, n, NULL, NULL);
  result_jpvt->shape[0] = n;

  switch(matrix->dtype) {
    case nm_bool:
    {
      //not supported error
      break;
    }
    case nm_int:
    {
      //not supported error
      break;
    }
    case nm_float32:
    {
      float* elements = ALLOC_N(float, matrix->count);
      memcpy(elements, matrix->elements, sizeof(float)*matrix->count);
      float* tau_elements = ALLOC_N(float, result_tau->count);
      int* jpvt_elements = ALLOC_N(int, result_jpvt->count);
      info = LAPACKE_sgeqp3(LAPACK_ROW_MAJOR, m, n, elements, lda, jpvt_elements, tau_elements);

      result_qr->elements = elements;
      result_tau->elements = tau_elements;
      result_jpvt->elements = jpvt_elements;

      VALUE qr = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_qr);
      VALUE tau = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_tau);
      VALUE jpvt = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_jpvt);
      return rb_ary_new3(3, qr, tau, jpvt);
      break;
    }
    case nm_float64:
    {
      double* elements = ALLOC_N(double, matrix->count);
      memcpy(elements, matrix->elements, sizeof(double)*matrix->count);
      double* tau_elements = ALLOC_N(double, result_tau->count);
      int* jpvt_elements = ALLOC_N(int, result_jpvt->count);
      info = LAPACKE_dgeqp3(LAPACK_ROW_MAJOR, m, n, elements, lda, jpvt_elements, tau_elements);

      result_qr->elements = elements;
      result_tau->elements = tau_elements;
      result_jpvt->elements = jpvt_elements;

      VALUE qr = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_qr);
      VALUE tau = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_tau);
      VALUE jpvt = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_jpvt);
      return rb_ary_new3(3, qr, tau, jpvt);
      break;
    }
    case nm_complex32:
    {
      float complex* elements = ALLOC_N(float complex, matrix->count);
      memcpy(elements, matrix->elements, sizeof(float complex)*matrix->count);
      float complex* tau_elements = ALLOC_N(float complex, result_tau->count);
      int* jpvt_elements = ALLOC_N(int, result_jpvt->count);
      info = LAPACKE_cgeqp3(LAPACK_ROW_MAJOR, m, n, elements, lda, jpvt_elements, tau_elements);

      result_qr->elements = elements;
      result_tau->elements = tau_elements;
      result_jpvt->elements = jpvt_elements;

      VALUE qr = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_qr);
      VALUE tau = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_tau);
      VALUE jpvt = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_jpvt);
      return rb_ary_new3(3, qr, tau, jpvt);
      break;
    }
    case nm_complex64:
    {
      double complex* elements = ALLOC_N(double complex, matrix->count);
      memcpy(elements, matrix->elements, sizeof(double complex)*matrix->count);
      double complex* tau_elements = ALLOC_N(double complex, result_tau->count);
      int* jpvt_elements = ALLOC_N(int, result_jpvt->count);
      info = LAPACKE_zgeqp3(LAPACK_ROW_MAJOR, m, n, elements, lda, jpvt_elements, tau_elements);

      result_qr->elements = elements;
      result_tau->elements = tau_elements;
      result_jpvt->elements = jpvt_elements;

      VALUE qr = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_qr);
      VALUE tau = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_tau);
      VALUE jpvt = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_jpvt);
      return rb_ary_new3(3, qr, tau, jpvt);
      break;
    }
  }
  return INT2NUM(-1);
}

/*
 * POTRF computes the Cholesky factorization of a real symmetric
 * positive definite matrix A.
 * 
 * The factorization has the form
 *    A = U**T * U, if UPLO = 'U', or
 *    A = L * L**T, if UPLO = 'L',
 * where U is an upper triangular matrix and L is lower triangular.
 * 
 * This is the block version of the algorithm, calling Level 3 BLAS.
 *
 */
VALUE nm_potrf(int argc, VALUE* argv) {
  nmatrix* matrix;
  Data_Get_Struct(argv[0], nmatrix, matrix);

  bool lower = (bool)RTEST(argv[1]);

  int m = matrix->shape[0]; //no. of rows
  int n = matrix->shape[1]; //no. of cols
  int lda = n, info = -1;
  char uplo = lower ? 'L' : 'U';

  nmatrix* result_cho = nmatrix_new(matrix->dtype, matrix->stype, 2, matrix->count, matrix->shape, NULL);

  switch(matrix->dtype) {
    case nm_bool:
    {
      //not supported error
      break;
    }
    case nm_int:
    {
      //not supported error
      break;
    }
    case nm_float32:
    {
      float* elements = ALLOC_N(float, matrix->count);
      memcpy(elements, matrix->elements, sizeof(float)*matrix->count);
      info = LAPACKE_spotrf(LAPACK_ROW_MAJOR, uplo, n, elements, lda);

      result_cho->elements = elements;

      return Data_Wrap_Struct(NMatrix, NULL, nm_free, result_cho);
      break;
    }
    case nm_float64:
    {
      double* elements = ALLOC_N(double, matrix->count);
      memcpy(elements, matrix->elements, sizeof(double)*matrix->count);
      info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, uplo, n, elements, lda);

      result_cho->elements = elements;

      return Data_Wrap_Struct(NMatrix, NULL, nm_free, result_cho);
      break;
    }
    case nm_complex32:
    {
      float complex* elements = ALLOC_N(float complex, matrix->count);
      memcpy(elements, matrix->elements, sizeof(float complex)*matrix->count);
      info = LAPACKE_cpotrf(LAPACK_ROW_MAJOR, uplo, n, elements, lda);

      result_cho->elements = elements;

      return Data_Wrap_Struct(NMatrix, NULL, nm_free, result_cho);
      break;
    }
    case nm_complex64:
    {
      double complex* elements = ALLOC_N(double complex, matrix->count);
      memcpy(elements, matrix->elements, sizeof(double complex)*matrix->count);
      info = LAPACKE_zpotrf(LAPACK_ROW_MAJOR, uplo, n, elements, lda);

      result_cho->elements = elements;

      return Data_Wrap_Struct(NMatrix, NULL, nm_free, result_cho);
      break;
    }
  }
  return INT2NUM(-1);
}

/*
 * POTRS solves a system of linear equations A*X = B with a symmetric
 * positive definite matrix A using the Cholesky factorization
 * A = U**T*U or A = L*L**T computed by POTRF.
 *
 */
VALUE nm_potrs(int argc, VALUE* argv) {
  nmatrix* matrix_a;
  Data_Get_Struct(argv[0], nmatrix, matrix_a);

  int m_a = matrix_a->shape[0]; //no. of rows
  int n_a = matrix_a->shape[1]; //no. of cols
  int lda_a = n_a, info = -1;

  nmatrix* matrix_b;
  Data_Get_Struct(argv[1], nmatrix, matrix_b);

  int m_b = matrix_b->shape[0]; //no. of rows
  int n_b = 1; //no. of cols
  int lda_b = n_b;

  bool lower = (bool)RTEST(argv[2]);
  char uplo = lower ? 'L' : 'U';

  nmatrix* result = nmatrix_new(matrix_b->dtype, matrix_b->stype, 1, matrix_b->count, matrix_b->shape, NULL);

  switch(matrix_a->dtype) {
    case nm_bool:
    {
      //not supported error
      break;
    }
    case nm_int:
    {
      //not supported error
      break;
    }
    case nm_float32:
    {
      float* elements_a = (float*)matrix_a->elements;
      float* elements_b = ALLOC_N(float, matrix_b->count);
      memcpy(elements_b, matrix_b->elements, sizeof(float)*matrix_b->count);
      info = LAPACKE_spotrs(LAPACK_ROW_MAJOR, uplo, n_a, n_b, elements_a, lda_a, elements_b, lda_b);

      result->elements = elements_b;

      return Data_Wrap_Struct(NMatrix, NULL, nm_free, result);
      break;
    }
    case nm_float64:
    {
      double* elements_a = (double*)matrix_a->elements;
      double* elements_b = ALLOC_N(double, matrix_b->count);
      memcpy(elements_b, matrix_b->elements, sizeof(double)*matrix_b->count);
      info = LAPACKE_dpotrs(LAPACK_ROW_MAJOR, uplo, n_a, n_b, elements_a, lda_a, elements_b, lda_b);

      result->elements = elements_b;

      return Data_Wrap_Struct(NMatrix, NULL, nm_free, result);
      break;
    }
    case nm_complex32:
    {
      float complex* elements_a = (float complex*)matrix_a->elements;
      float complex* elements_b = ALLOC_N(float complex, matrix_b->count);
      memcpy(elements_b, matrix_b->elements, sizeof(float complex)*matrix_b->count);
      info = LAPACKE_cpotrs(LAPACK_ROW_MAJOR, uplo, n_a, n_b, elements_a, lda_a, elements_b, lda_b);

      result->elements = elements_b;

      return Data_Wrap_Struct(NMatrix, NULL, nm_free, result);
      break;
    }
    case nm_complex64:
    {
      double complex* elements_a = (double complex*)matrix_a->elements;
      double complex* elements_b = ALLOC_N(double complex, matrix_b->count);
      memcpy(elements_b, matrix_b->elements, sizeof(double complex)*matrix_b->count);
      info = LAPACKE_zpotrs(LAPACK_ROW_MAJOR, uplo, n_a, n_b, elements_a, lda_a, elements_b, lda_b);

      result->elements = elements_b;

      return Data_Wrap_Struct(NMatrix, NULL, nm_free, result);
      break;
    }
  }
  return INT2NUM(-1);
}

/*
 * GESDD computes the singular value decomposition (SVD) of a real
 * M-by-N matrix A, optionally computing the left and right singular
 * vectors. If singular vectors are desired, it uses a
 * divide-and-conquer algorithm.
 *
 * The SVD is written
 *    A = U * SIGMA * transpose(V)
 *
 * where SIGMA is an M-by-N matrix which is zero except for its
 * min(m,n) diagonal elements, U is an M-by-M orthogonal matrix, and
 * V is an N-by-N orthogonal matrix. The diagonal elements of SIGMA
 * are the singular values of A; they are real and non-negative, and
 * are returned in descending order. The first min(m,n) columns of
 * U and V are the left and right singular vectors of A.
 * 
 * Note that the routine returns VT = V**T, not V.
 * 
 * The divide and conquer algorithm makes very mild assumptions about
 * floating point arithmetic. It will work on machines with a guard
 * digit in add/subtract, or on those binary machines without guard
 * digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
 * Cray-2. It could conceivably fail on hexadecimal or decimal machines
 * without guard digits, but we know of none.
 * 
 */
VALUE nm_gesdd(int argc, VALUE* argv) {
  nmatrix* matrix;
  Data_Get_Struct(argv[0], nmatrix, matrix);

  int m = matrix->shape[0]; //no. of rows
  int n = matrix->shape[1]; //no. of cols
  int lda = n, info = -1;

  bool full_matrices = RTEST(argv[1]);
  bool compute_uv = RTEST(argv[2]);

  char jobz = full_matrices ? 'A' : 'S';

  if(compute_uv == false) {
    jobz = 'N';
  }

  nmatrix* result = nmatrix_new(matrix->dtype, matrix->stype, 2, matrix->count, matrix->shape, NULL);

  switch(matrix->dtype) {
    case nm_bool:
    {
      //not supported error
      break;
    }
    case nm_int:
    {
      //not supported error
      break;
    }
    case nm_float32:
    {
      break;
    }
    case nm_float64:
    {
      // lapack_int LAPACKE_dgesdd( int matrix_layout, char jobz, lapack_int m,
      //                      lapack_int n, double* a, lapack_int lda, double* s,
      //                      double* u, lapack_int ldu, double* vt,
      //                      lapack_int ldvt );
      double* elements = (double*)matrix->elements;
      double* elements_s;
      double* elements_u;
      //info = LAPACKE_dgesdd(LAPACK_ROW_MAJOR, jobz, m, n, elements, lda, elements_s, elements_u);
      break;
    }
    case nm_complex32:
    {
      break;
    }
    case nm_complex64:
    {
      break;
    }
  }
  return INT2NUM(-1);
}

/*
 * GETRF computes an LU factorization of a general M-by-N matrix A
 * using partial pivoting with row interchanges.
 * 
 * The factorization has the form
 *    A = P * L * U
 * where P is a permutation matrix, L is lower triangular with unit
 * diagonal elements (lower trapezoidal if m > n), and U is upper
 * triangular (upper trapezoidal if m < n).
 * 
 * This is the right-looking Level 3 BLAS version of the algorithm.
 *
 */
VALUE nm_getrf(int argc, VALUE* argv) {
  nmatrix* matrix;
  Data_Get_Struct(argv[0], nmatrix, matrix);

  int m = matrix->shape[0]; //no. of rows
  int n = matrix->shape[1]; //no. of cols
  int lda = n, info = -1;

  nmatrix* result_lu = nmatrix_new(matrix->dtype, matrix->stype, 2, matrix->count, matrix->shape, NULL);
  nmatrix* result_ipiv = nmatrix_new(matrix->dtype, matrix->stype, 1, min(m, n), NULL, NULL);
  result_ipiv->shape[0] = min(m, n);

  switch(matrix->dtype) {
    case nm_bool:
    {
      //not supported error
      break;
    }
    case nm_int:
    {
      //not supported error
      break;
    }
    case nm_float32:
    {
      float* elements = ALLOC_N(float, matrix->count);
      memcpy(elements, matrix->elements, sizeof(float)*matrix->count);
      float* ipiv_elements = ALLOC_N(int, min(m, n));
      info = LAPACKE_sgetrf(LAPACK_ROW_MAJOR, m, n, elements, lda, ipiv_elements);

      result_lu->elements = elements;
      result_ipiv->elements = ipiv_elements;

      VALUE lu = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_lu);
      VALUE ipiv = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_ipiv);
      return rb_ary_new3(2, lu, ipiv);
      break;
    }
    case nm_float64:
    {
      double* elements = ALLOC_N(double, matrix->count);
      memcpy(elements, matrix->elements, sizeof(double)*matrix->count);
      double* ipiv_elements = ALLOC_N(int, min(m, n));
      info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, m, n, elements, lda, ipiv_elements);

      result_lu->elements = elements;
      result_ipiv->elements = ipiv_elements;

      VALUE lu = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_lu);
      VALUE ipiv = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_ipiv);
      return rb_ary_new3(2, lu, ipiv);
      break;
    }
    case nm_complex32:
    {
      float complex* elements = ALLOC_N(float complex, matrix->count);
      memcpy(elements, matrix->elements, sizeof(float complex)*matrix->count);
      float complex* ipiv_elements = ALLOC_N(int, min(m, n));
      info = LAPACKE_cgetrf(LAPACK_ROW_MAJOR, m, n, elements, lda, ipiv_elements);

      result_lu->elements = elements;
      result_ipiv->elements = ipiv_elements;

      VALUE lu = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_lu);
      VALUE ipiv = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_ipiv);
      return rb_ary_new3(2, lu, ipiv);
      break;
    }
    case nm_complex64:
    {
      double complex* elements = ALLOC_N(double complex, matrix->count);
      memcpy(elements, matrix->elements, sizeof(double complex)*matrix->count);
      double complex* ipiv_elements = ALLOC_N(int, min(m, n));
      info = LAPACKE_zgetrf(LAPACK_ROW_MAJOR, m, n, elements, lda, ipiv_elements);

      result_lu->elements = elements;
      result_ipiv->elements = ipiv_elements;

      VALUE lu = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_lu);
      VALUE ipiv = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_ipiv);
      return rb_ary_new3(2, lu, ipiv);
      break;
    }
  }
  return INT2NUM(-1);
}

/*
 * GETRS solves a system of linear equations
 *    A * X = B  or  A**T * X = B
 * with a general N-by-N matrix A using the LU factorization computed
 * by GETRF.
 *
 */
VALUE nm_getrs(int argc, VALUE* argv) {
  nmatrix* matrix_a;
  Data_Get_Struct(argv[0], nmatrix, matrix_a);

  int m_a = matrix_a->shape[0]; //no. of rows
  int n_a = matrix_a->shape[1]; //no. of cols
  int lda_a = n_a, info = -1;

  nmatrix* matrix_ipiv;
  Data_Get_Struct(argv[1], nmatrix, matrix_ipiv);

  nmatrix* matrix_b;
  Data_Get_Struct(argv[2], nmatrix, matrix_b);

  int m_b = matrix_b->shape[0]; //no. of rows
  int n_b = 1; //no. of cols
  int lda_b = n_b;

  int tra = NUM2INT(argv[3]);
  char trans = 'N';

  if(tra == 1) {
    trans = 'T';
  }
  else if(tra == 2) {
    trans = 'C';
  }

  nmatrix* result = nmatrix_new(matrix_b->dtype, matrix_b->stype, 1, matrix_b->count, matrix_b->shape, NULL);

  switch(matrix_a->dtype) {
    case nm_bool:
    {
      //not supported error
      break;
    }
    case nm_int:
    {
      //not supported error
      break;
    }
    case nm_float32:
    {
      float* elements_a = (float*)matrix_a->elements;
      float* elements_b = ALLOC_N(float, matrix_b->count);
      memcpy(elements_b, matrix_b->elements, sizeof(float)*matrix_b->count);
      int* ipiv = (int*)matrix_ipiv->elements;
      info = LAPACKE_sgetrs(LAPACK_ROW_MAJOR, trans, n_a, n_b, elements_a, lda_a, ipiv, elements_b, lda_b);

      result->elements = elements_b;

      return Data_Wrap_Struct(NMatrix, NULL, nm_free, result);
      break;
    }
    case nm_float64:
    {
      double* elements_a = (double*)matrix_a->elements;
      double* elements_b = ALLOC_N(double, matrix_b->count);
      memcpy(elements_b, matrix_b->elements, sizeof(double)*matrix_b->count);
      int* ipiv = (int*)matrix_ipiv->elements;
      info = LAPACKE_dgetrs(LAPACK_ROW_MAJOR, trans, n_a, n_b, elements_a, lda_a, ipiv, elements_b, lda_b);

      result->elements = elements_b;

      return Data_Wrap_Struct(NMatrix, NULL, nm_free, result);
      break;
    }
    case nm_complex32:
    {
      float complex* elements_a = (float complex*)matrix_a->elements;
      float complex* elements_b = ALLOC_N(float complex, matrix_b->count);
      memcpy(elements_b, matrix_b->elements, sizeof(float complex)*matrix_b->count);
      int* ipiv = (int*)matrix_ipiv->elements;
      info = LAPACKE_cgetrs(LAPACK_ROW_MAJOR, trans, n_a, n_b, elements_a, lda_a, ipiv, elements_b, lda_b);

      result->elements = elements_b;

      return Data_Wrap_Struct(NMatrix, NULL, nm_free, result);
      break;
    }
    case nm_complex64:
    {
      double complex* elements_a = (double complex*)matrix_a->elements;
      double complex* elements_b = ALLOC_N(double complex, matrix_b->count);
      memcpy(elements_b, matrix_b->elements, sizeof(double complex)*matrix_b->count);
      int* ipiv = (int*)matrix_ipiv->elements;
      info = LAPACKE_zgetrs(LAPACK_ROW_MAJOR, trans, n_a, n_b, elements_a, lda_a, ipiv, elements_b, lda_b);

      result->elements = elements_b;

      return Data_Wrap_Struct(NMatrix, NULL, nm_free, result);
      break;
    }
  }
  return INT2NUM(-1);
}

/*
 * GETRI computes the inverse of a matrix using the LU factorization
 * computed by GETRF.
 * 
 * This method inverts U and then computes inv(A) by solving the system
 * inv(A)*L = inv(U) for inv(A).
 *
 */
VALUE nm_getri(int argc, VALUE* argv) {
  nmatrix* matrix_lu;
  Data_Get_Struct(argv[0], nmatrix, matrix_lu);

  int m = matrix_lu->shape[0]; //no. of rows
  int n = matrix_lu->shape[1]; //no. of cols
  int lda = n, info = -1;

  nmatrix* matrix_ipiv;
  Data_Get_Struct(argv[1], nmatrix, matrix_ipiv);

  nmatrix* result = nmatrix_new(matrix_lu->dtype, matrix_lu->stype, 2, matrix_lu->count, matrix_lu->shape, NULL);

  switch(matrix_lu->dtype) {
    case nm_bool:
    {
      //not supported error
      break;
    }
    case nm_int:
    {
      //not supported error
      break;
    }
    case nm_float32:
    {
      float* elements = ALLOC_N(float, matrix_lu->count);
      memcpy(elements, matrix_lu->elements, sizeof(float)*matrix_lu->count);
      int* elements_ipiv = (int*)matrix_ipiv->elements;
      info = LAPACKE_sgetri(LAPACK_ROW_MAJOR, n, elements, lda, elements_ipiv);

      result->elements = elements;

      return Data_Wrap_Struct(NMatrix, NULL, nm_free, result);
      break;
    }
    case nm_float64:
    {
      double* elements = ALLOC_N(double, matrix_lu->count);
      memcpy(elements, matrix_lu->elements, sizeof(double)*matrix_lu->count);
      int* elements_ipiv = (int*)matrix_ipiv->elements;
      info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, elements, lda, elements_ipiv);

      result->elements = elements;

      return Data_Wrap_Struct(NMatrix, NULL, nm_free, result);
      break;
    }
    case nm_complex32:
    {
      float complex* elements = ALLOC_N(float complex, matrix_lu->count);
      memcpy(elements, matrix_lu->elements, sizeof(float complex)*matrix_lu->count);
      int* elements_ipiv = (int*)matrix_ipiv->elements;
      info = LAPACKE_cgetri(LAPACK_ROW_MAJOR, n, elements, lda, elements_ipiv);

      result->elements = elements;

      return Data_Wrap_Struct(NMatrix, NULL, nm_free, result);
      break;
    }
    case nm_complex64:
    {
      double complex* elements = ALLOC_N(double complex, matrix_lu->count);
      memcpy(elements, matrix_lu->elements, sizeof(double complex)*matrix_lu->count);
      int* elements_ipiv = (int*)matrix_ipiv->elements;
      info = LAPACKE_zgetri(LAPACK_ROW_MAJOR, n, elements, lda, elements_ipiv);

      result->elements = elements;

      return Data_Wrap_Struct(NMatrix, NULL, nm_free, result);
      break;
    }
  }
  return INT2NUM(-1);
}

/*
 * GELSS computes the minimum norm solution to a real linear least
 * squares problem:
 *
 * Minimize 2-norm(| b - A*x |).
 *
 * using the singular value decomposition (SVD) of A. A is an M-by-N
 * matrix which may be rank-deficient.
 *
 * Several right hand side vectors b and solution vectors x can be
 * handled in a single call; they are stored as the columns of the
 * M-by-NRHS right hand side matrix B and the N-by-NRHS solution matrix
 * X.
 *
 * The effective rank of A is determined by treating as zero those
 * singular values which are less than RCOND times the largest singular
 * value.
 *
 */
VALUE nm_gelss(int argc, VALUE* argv) {
  //TODO
  return INT2NUM(-1);
}

/*
 * POSV computes the solution to a real system of linear equations
 *    A * X = B,
 * where A is an N-by-N symmetric positive definite matrix and X and B
 * are N-by-NRHS matrices.
 *
 * The Cholesky decomposition is used to factor A as
 *    A = U**T* U,  if UPLO = 'U', or
 *    A = L * L**T,  if UPLO = 'L',
 * where U is an upper triangular matrix and L is a lower triangular
 * matrix. The factored form of A is then used to solve the system of
 * equations A * X = B.
 *
 */
VALUE nm_posv(int argc, VALUE* argv) {
  nmatrix* matrix_a;
  Data_Get_Struct(argv[0], nmatrix, matrix_a);

  int m_a = matrix_a->shape[0]; //no. of rows
  int n_a = matrix_a->shape[1]; //no. of cols
  int lda_a = n_a, info = -1;

  nmatrix* matrix_b;
  Data_Get_Struct(argv[1], nmatrix, matrix_b);

  int m_b = matrix_b->shape[0]; //no. of rows
  int n_b = matrix_b->shape[1]; //no. of cols
  int lda_b = n_b;

  bool lower = (bool)RTEST(argv[2]);
  char uplo = lower ? 'L' : 'U';

  nmatrix* result_c = nmatrix_new(matrix_a->dtype, matrix_a->stype, 2, matrix_a->count, matrix_a->shape, NULL);
  nmatrix* result_x = nmatrix_new(matrix_b->dtype, matrix_b->stype, 2, matrix_b->count, matrix_b->shape, NULL);

  switch(matrix_a->dtype) {
    case nm_bool:
    {
      //not supported error
      break;
    }
    case nm_int:
    {
      //not supported error
      break;
    }
    case nm_float32:
    {
      float* elements_a = ALLOC_N(float, matrix_a->count);
      memcpy(elements_a, matrix_a->elements, sizeof(float)*matrix_a->count);
      float* elements_b = ALLOC_N(float, matrix_b->count);
      memcpy(elements_b, matrix_b->elements, sizeof(float)*matrix_b->count);
      info = LAPACKE_sposv(LAPACK_ROW_MAJOR, uplo, n_a, n_b, elements_a, lda_a, elements_b, lda_b);

      result_c->elements = elements_a;
      result_x->elements = elements_b;

      VALUE c = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_c);
      VALUE x = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_x);
      return rb_ary_new3(2, c, x);
      break;
    }
    case nm_float64:
    {
      double* elements_a = ALLOC_N(double, matrix_a->count);
      memcpy(elements_a, matrix_a->elements, sizeof(double)*matrix_a->count);
      double* elements_b = ALLOC_N(double, matrix_b->count);
      memcpy(elements_b, matrix_b->elements, sizeof(double)*matrix_b->count);
      info = LAPACKE_dposv(LAPACK_ROW_MAJOR, uplo, n_a, n_b, elements_a, lda_a, elements_b, lda_b);

      result_c->elements = elements_a;
      result_x->elements = elements_b;

      VALUE c = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_c);
      VALUE x = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_x);
      return rb_ary_new3(2, c, x);
      break;
    }
    case nm_complex32:
    {
      float complex* elements_a = ALLOC_N(float complex, matrix_a->count);
      memcpy(elements_a, matrix_a->elements, sizeof(float complex)*matrix_a->count);
      float complex* elements_b = ALLOC_N(float complex, matrix_b->count);
      memcpy(elements_b, matrix_b->elements, sizeof(float complex)*matrix_b->count);
      info = LAPACKE_cposv(LAPACK_ROW_MAJOR, uplo, n_a, n_b, elements_a, lda_a, elements_b, lda_b);

      result_c->elements = elements_a;
      result_x->elements = elements_b;

      VALUE c = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_c);
      VALUE x = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_x);
      return rb_ary_new3(2, c, x);
      break;
    }
    case nm_complex64:
    {
      double complex* elements_a = ALLOC_N(double complex, matrix_a->count);
      memcpy(elements_a, matrix_a->elements, sizeof(double complex)*matrix_a->count);
      double complex* elements_b = ALLOC_N(double complex, matrix_b->count);
      memcpy(elements_b, matrix_b->elements, sizeof(double complex)*matrix_b->count);
      info = LAPACKE_zposv(LAPACK_ROW_MAJOR, uplo, n_a, n_b, elements_a, lda_a, elements_b, lda_b);

      result_c->elements = elements_a;
      result_x->elements = elements_b;

      VALUE c = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_c);
      VALUE x = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_x);
      return rb_ary_new3(2, c, x);
      break;
    }
  }
  return INT2NUM(-1);
}

/*
 * GESV computes the solution to a real system of linear equations
 *    A * X = B,
 * where A is an N-by-N matrix and X and B are N-by-NRHS matrices.
 *
 * The LU decomposition with partial pivoting and row interchanges is
 * used to factor A as
 *    A = P * L * U,
 * where P is a permutation matrix, L is unit lower triangular, and U is
 * upper triangular. The factored form of A is then used to solve the
 * system of equations A * X = B.
 *
 */
VALUE nm_gesv(int argc, VALUE* argv) {
  nmatrix* matrix_a;
  Data_Get_Struct(argv[0], nmatrix, matrix_a);

  int m_a = matrix_a->shape[0]; //no. of rows
  int n_a = matrix_a->shape[1]; //no. of cols
  int lda_a = n_a, info = -1;

  nmatrix* matrix_b;
  Data_Get_Struct(argv[1], nmatrix, matrix_b);

  int m_b = matrix_b->shape[0]; //no. of rows
  int n_b = matrix_b->shape[1]; //no. of cols
  int lda_b = n_b;

  bool lower = (bool)RTEST(argv[2]);
  char uplo = lower ? 'L' : 'U';

  nmatrix* result_lu = nmatrix_new(matrix_a->dtype, matrix_a->stype, 2, matrix_a->count, matrix_a->shape, NULL);
  nmatrix* result_x = nmatrix_new(matrix_b->dtype, matrix_b->stype, 2, matrix_b->count, matrix_b->shape, NULL);
  nmatrix* result_ipiv = nmatrix_new(nm_int, matrix_a->stype, 1, n_a, NULL, NULL);
  result_ipiv->shape[0] = n_a;

  switch(matrix_a->dtype) {
    case nm_bool:
    {
      //not supported error
      break;
    }
    case nm_int:
    {
      //not supported error
      break;
    }
    case nm_float32:
    {
      int* ipiv_elements = ALLOC_N(int, n_a);
      float* elements_a = ALLOC_N(float, matrix_a->count);
      memcpy(elements_a, matrix_a->elements, sizeof(float)*matrix_a->count);
      float* elements_b = ALLOC_N(float, matrix_b->count);
      memcpy(elements_b, matrix_b->elements, sizeof(float)*matrix_b->count);
      info = LAPACKE_sgesv(LAPACK_ROW_MAJOR, n_a, n_b, elements_a, lda_a, ipiv_elements, elements_b, lda_b);

      result_lu->elements = elements_a;
      result_x->elements = elements_b;
      result_ipiv->elements = ipiv_elements;

      VALUE lu = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_lu);
      VALUE x = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_x);
      VALUE ipiv = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_ipiv);
      return rb_ary_new3(3, lu, x, ipiv);
      break;
    }
    case nm_float64:
    {
      int* ipiv_elements = ALLOC_N(int, n_a);
      double* elements_a = ALLOC_N(double, matrix_a->count);
      memcpy(elements_a, matrix_a->elements, sizeof(double)*matrix_a->count);
      double* elements_b = ALLOC_N(double, matrix_b->count);
      memcpy(elements_b, matrix_b->elements, sizeof(double)*matrix_b->count);
      info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n_a, n_b, elements_a, lda_a, ipiv_elements, elements_b, lda_b);

      result_lu->elements = elements_a;
      result_x->elements = elements_b;
      result_ipiv->elements = ipiv_elements;

      VALUE lu = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_lu);
      VALUE x = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_x);
      VALUE ipiv = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_ipiv);
      return rb_ary_new3(3, lu, x, ipiv);
      break;
    }
    case nm_complex32:
    {
      int* ipiv_elements = ALLOC_N(int, n_a);
      float complex* elements_a = ALLOC_N(float complex, matrix_a->count);
      memcpy(elements_a, matrix_a->elements, sizeof(float complex)*matrix_a->count);
      float complex* elements_b = ALLOC_N(float complex, matrix_b->count);
      memcpy(elements_b, matrix_b->elements, sizeof(float complex)*matrix_b->count);
      info = LAPACKE_cgesv(LAPACK_ROW_MAJOR, n_a, n_b, elements_a, lda_a, ipiv_elements, elements_b, lda_b);

      result_lu->elements = elements_a;
      result_x->elements = elements_b;
      result_ipiv->elements = ipiv_elements;

      VALUE lu = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_lu);
      VALUE x = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_x);
      VALUE ipiv = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_ipiv);
      return rb_ary_new3(3, lu, x, ipiv);
      break;
    }
    case nm_complex64:
    {
      int* ipiv_elements = ALLOC_N(int, n_a);
      double complex* elements_a = ALLOC_N(double complex, matrix_a->count);
      memcpy(elements_a, matrix_a->elements, sizeof(double complex)*matrix_a->count);
      double complex* elements_b = ALLOC_N(double complex, matrix_b->count);
      memcpy(elements_b, matrix_b->elements, sizeof(double complex)*matrix_b->count);
      info = LAPACKE_zgesv(LAPACK_ROW_MAJOR, n_a, n_b, elements_a, lda_a, ipiv_elements, elements_b, lda_b);

      result_lu->elements = elements_a;
      result_x->elements = elements_b;
      result_ipiv->elements = ipiv_elements;

      VALUE lu = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_lu);
      VALUE x = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_x);
      VALUE ipiv = Data_Wrap_Struct(NMatrix, NULL, nm_free, result_ipiv);
      return rb_ary_new3(3, lu, x, ipiv);
      break;
    }
  }
  return INT2NUM(-1);
}

/*
 * LANGE returns the value of the one norm, or the Frobenius norm, or
 * the infinity norm, or the element of largest absolute value of a
 * real matrix A.
 *
 */
VALUE nm_lange(int argc, VALUE* argv) {
  nmatrix* matrix;
  Data_Get_Struct(argv[0], nmatrix, matrix);

  int m = matrix->shape[0]; //no. of rows
  int n = matrix->shape[1]; //no. of cols
  int lda = n;

  char norm = NUM2CHAR(argv[1]);

  switch(matrix->dtype) {
    case nm_bool:
    {
      //not supported error
      break;
    }
    case nm_int:
    {
      //not supported error
      break;
    }
    case nm_float32:
    {
      float val = LAPACKE_slange(LAPACK_ROW_MAJOR, norm, m, n, matrix->elements, lda);
      return val;
      break;
    }
    case nm_float64:
    {
      double val = LAPACKE_dlange(LAPACK_ROW_MAJOR, norm, m, n, matrix->elements, lda);
      return val;
      break;
    }
    case nm_complex32:
    {
      float complex val = LAPACKE_clange(LAPACK_ROW_MAJOR, norm, m, n, matrix->elements, lda);
      return val;
      break;
    }
    case nm_complex64:
    {
      double complex val = LAPACKE_zlange(LAPACK_ROW_MAJOR, norm, m, n, matrix->elements, lda);
      return val;
      break;
    }
  }
  return INT2NUM(-1);
}

// TODO: m should represent no. of rows and n no. of cols throughout

/*
 *	Calculates matrix inverse.
 *	Args:
 *	-	self matrix, type: NMatrix
 *	
 *	returns the inverse matrix of type NMatrix
 */
VALUE nm_invert(VALUE self){
  nmatrix* matrix;
  Data_Get_Struct(self, nmatrix, matrix);

  nmatrix* result = ALLOC(nmatrix);
  result->dtype = matrix->dtype;
  result->stype = matrix->stype;
  result->ndims = matrix->ndims;
  result->shape = ALLOC_N(size_t, result->ndims);

  result->shape[0] =  matrix->shape[0];
  result->shape[1] =  matrix->shape[1];
  result->count = matrix->count;
  double* elements = ALLOC_N(double, result->shape[0] * result->shape[1]);


  int n = (int)matrix->shape[1];
  int m = (int)matrix->shape[0];
  int* ipiv = ALLOC_N(int, min(m,n)+1);
  dgetrf(matrix->elements, n, m, ipiv, elements);
  int lda = n;

  LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, elements, lda, ipiv);
  result->elements = elements;
  return Data_Wrap_Struct(NMatrix, NULL, nm_free, result);
}

void sgetrf(const float* arr, const size_t cols, const size_t rows, int* ipiv, float* arr2) {
  int m = (int)cols;
  int n = (int)rows;
  memcpy(arr2, arr, sizeof(float)*rows*cols);
  int lda = m;
  LAPACKE_sgetrf(LAPACK_ROW_MAJOR,m,n,arr2,lda,ipiv);
}

void dgetrf(const double* arr, const size_t cols, const size_t rows, int* ipiv, double* arr2) {
  int m = (int)cols;
  int n = (int)rows;
  memcpy(arr2, arr, sizeof(double)*rows*cols);
  int lda = m;
  LAPACKE_dgetrf(LAPACK_ROW_MAJOR,m,n,arr2,lda,ipiv);
}

void cgetrf(const float complex* arr, const size_t cols, const size_t rows, int* ipiv, float complex* arr2) {
  int m = (int)cols;
  int n = (int)rows;
  memcpy(arr2, arr, sizeof(float complex)*rows*cols);
  int lda = m;
  LAPACKE_cgetrf(LAPACK_ROW_MAJOR,m,n,arr2,lda,ipiv);
}

void zgetrf(const double complex* arr, const size_t cols, const size_t rows, int* ipiv, double complex* arr2) {
  int m = (int)cols;
  int n = (int)rows;
  memcpy(arr2, arr, sizeof(double complex)*rows*cols);
  int lda = m;
  LAPACKE_zgetrf(LAPACK_ROW_MAJOR,m,n,arr2,lda,ipiv);
}

/*
 *	Solves a system of linear equations.
 *	Args:
 *	-	lhs matrix (square), type: NMatrix
 *	-	rhs vector, type: NMatrix
 *	
 *	returns the vector of type NMatrix with values of unknowns
 */
VALUE nm_solve(VALUE self, VALUE rhs_val){
  nmatrix* lhs;
  nmatrix* rhs;
  Data_Get_Struct(self, nmatrix, lhs);
  Data_Get_Struct(rhs_val, nmatrix, rhs);

  double* lhs_elements = ALLOC_N(double, lhs->count);
  memcpy(lhs_elements, lhs->elements, sizeof(double)*lhs->count);
  double* rhs_elements = ALLOC_N(double, rhs->count);
  memcpy(rhs_elements, rhs->elements, sizeof(double)*rhs->count);

  int n = (int)lhs->shape[1];
  //assert square matrix
  int nrhs = (int)rhs->shape[1];
  int lda = (int)lhs->shape[1];
  int ldb = (int)rhs->shape[1];
  int* ipiv = ALLOC_N(int, max(1,n));

  LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, lhs_elements,lda, ipiv, rhs_elements,ldb);

  nmatrix* result = ALLOC(nmatrix);
  result->dtype = rhs->dtype;
  result->stype = rhs->stype;
  result->ndims = rhs->ndims;
  result->shape = ALLOC_N(size_t, result->ndims);

  result->shape[0] =  rhs->shape[0];
  result->shape[1] =  rhs->shape[1];
  result->count = rhs->count;

  result->elements = rhs_elements;

  return Data_Wrap_Struct(NMatrix, NULL, nm_free, result);
}


/*
 *	Calculates matrix determinant.
 *	Args:
 *	-	matrix, type: NMatrix
 *	
 *	returns the determinant of matrix of type integer
 */
VALUE nm_det(VALUE self){
  nmatrix* matrix;
  Data_Get_Struct(self, nmatrix, matrix);

  int n = (int)matrix->shape[1];
  int m = (int)matrix->shape[0];
  
  double prod;
  
  switch (matrix->dtype) {
    case nm_float32:
    {
      
      float* elements = ALLOC_N(float, matrix->count);
      int* pivot = ALLOC_N(int, min(m,n)+1);
    
      sgetrf(matrix->elements, matrix->shape[1], matrix->shape[0], pivot,  elements);
    
      int num_perm = 0;
      int j = 0;
      for(int i = 0; i < min(m,n)+1; ++i){
        if(pivot[i]-1 != j){num_perm += 1;}
        j++;
      }
    
      prod = (num_perm % 2 == 1) ? 1 : -1;
    
      for(int i =0; i < min(m,n); i++){
        prod *= elements[matrix->shape[0]*i + i];
      }
      break;
    }
    case nm_float64:
    {

      double* elements = ALLOC_N(double, matrix->count);
      int* pivot = ALLOC_N(int, min(m,n)+1);
    
      dgetrf(matrix->elements, matrix->shape[1], matrix->shape[0], pivot, elements);
    
      int num_perm = 0;
      int j = 0;
      for(int i = 0; i < min(m,n)+1; ++i){
        if(pivot[i]-1 != j){num_perm += 1;}
        j++;
      }
    
      prod = (num_perm % 2 == 1) ? 1 : -1;
    
      for(int i =0; i < min(m,n); i++){
        prod *= elements[matrix->shape[0]*i + i];
      }
      break;
    }
  }
  return DBL2NUM(prod);
}

VALUE nm_least_square(VALUE self, VALUE rhs_val){
  nmatrix* lhs;
  nmatrix* rhs;
  Data_Get_Struct(self, nmatrix, lhs);
  Data_Get_Struct(rhs_val, nmatrix, rhs);

  int m = (int)lhs->shape[0];
  int n = (int)lhs->shape[1];
  int nrhs = (int)rhs->shape[1];
  int lda = (int)lhs->shape[1];
  int ldb = (int)rhs->shape[1];
  
  nmatrix* result = ALLOC(nmatrix);
  result->dtype = rhs->dtype;
  result->stype = rhs->stype;
  result->ndims = rhs->ndims;
  result->shape = ALLOC_N(size_t, result->ndims);

  result->shape[0] =  rhs->shape[0];
  result->shape[1] =  rhs->shape[1];
  result->count = rhs->count;

  // result->elements = rhs_elements;
  
  
  switch (lhs->dtype) {
    case nm_float32:
    {
      
      float* lhs_elements = ALLOC_N(float, lhs->count);
      memcpy(lhs_elements, lhs->elements, sizeof(float)*lhs->count);
    
      float* rhs_elements = ALLOC_N(float, rhs->count);
      memcpy(rhs_elements, rhs->elements, sizeof(float)*rhs->count);
      LAPACKE_sgels(LAPACK_ROW_MAJOR,'N',m,n,nrhs,lhs_elements,lda,rhs_elements,ldb);
      result->elements = rhs_elements;
      break;
    }
    case nm_float64:
    {
      
      double* lhs_elements = ALLOC_N(double, lhs->count);
      memcpy(lhs_elements, lhs->elements, sizeof(double)*lhs->count);
    
      double* rhs_elements = ALLOC_N(double, rhs->count);
      memcpy(rhs_elements, rhs->elements, sizeof(double)*rhs->count);
      
      LAPACKE_dgels(LAPACK_ROW_MAJOR,'N',m,n,nrhs,lhs_elements,lda,rhs_elements,ldb);
      result->elements = rhs_elements;
      break;
    }
  }

  //LAPACKE_dgels(LAPACK_ROW_MAJOR,'N',m,n,nrhs,lhs_elements,lda,rhs_elements,ldb);

  return Data_Wrap_Struct(NMatrix, NULL, nm_free, result);
}

VALUE nm_pinv(VALUE self){
  return Qnil;
}

VALUE nm_kronecker_prod(VALUE self){
  return Qnil;
}

VALUE nm_eig(VALUE self){
  return Qnil;
}

VALUE nm_eigh(VALUE self){
  return Qnil;
}

VALUE nm_eigvalsh(VALUE self){
  return Qnil;
}

// Decomposition

VALUE nm_lu(VALUE self){
  return Qnil;
}

VALUE nm_lu_factor(VALUE self){
  nmatrix* matrix;
  Data_Get_Struct(self, nmatrix, matrix);

  nmatrix* result_lu = ALLOC(nmatrix);
  result_lu->dtype = matrix->dtype;
  result_lu->stype = matrix->stype;
  result_lu->ndims = matrix->ndims;
  result_lu->shape = ALLOC_N(size_t, result_lu->ndims);

  result_lu->shape[0] = matrix->shape[0];
  result_lu->shape[1] = matrix->shape[1];
  result_lu->count = matrix->count;
  double* elements = ALLOC_N(double, result_lu->shape[0] * result_lu->shape[1]);


  int n = (int)matrix->shape[1];
  int m = (int)matrix->shape[0];
  nmatrix* result_piv = ALLOC(nmatrix);
  result_piv->dtype = nm_int;
  result_piv->stype = matrix->stype;
  result_piv->ndims = 1;
  result_piv->shape = ALLOC_N(size_t, result_piv->ndims);

  // TODO: confirm if length of ipiv is min(m, n) or min(m, n) + 1?
  result_piv->shape[0] = min(m,n);
  result_piv->count = min(m,n);
  int* ipiv = ALLOC_N(int, min(m,n));
  dgetrf(matrix->elements, matrix->shape[1], matrix->shape[0], ipiv, elements);

  result_lu->elements = elements;
  result_piv->elements = ipiv;

  VALUE ary = rb_ary_new();
  rb_ary_push(ary, Data_Wrap_Struct(NMatrix, NULL, nm_free, result_lu));
  rb_ary_push(ary, Data_Wrap_Struct(NMatrix, NULL, nm_free, result_piv));

  return ary;
}

VALUE nm_lu_solve(VALUE self, VALUE rhs_val){
  return Qnil;
}

VALUE nm_svd(VALUE self){
  return Qnil;
}

VALUE nm_svdvals(VALUE self){
  return Qnil;
}

VALUE nm_diagsvd(VALUE self){
  return Qnil;
}

VALUE nm_orth(VALUE self){
  return Qnil;
}

VALUE nm_cholesky(VALUE self){
  return Qnil;
}

VALUE nm_cholesky_solve(VALUE self){
  return Qnil;
}
