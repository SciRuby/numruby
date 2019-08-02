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
 *
 *
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

    }
    case nm_float64:
    {

    }
    case nm_complex32:
    {

    }
    case nm_complex64:
    {
      
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
