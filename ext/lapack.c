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

/*
 *  Computes the QR decomposition of matrix.
 *  Args:
 *  - input matrix, type: NMatrix
 *  - mode, type: String
 *  - pivoting, type: Boolean
 *  
 *  returns the vector of type NMatrix with values of unknowns
 */
VALUE nm_qr(VALUE self, VALUE mode, VALUE pivoting){
  nmatrix* matrix;
  Data_Get_Struct(self, nmatrix, matrix);

  rb_funcall(mode, rb_intern("to_lower"), 0);
  VALUE mode_full = rb_str_new2("full");
  VALUE mode_r = rb_str_new2("r");
  VALUE mode_economic = rb_str_new2("economic");
  VALUE mode_raw = rb_str_new2("raw");

  bool pivot = (bool)RTEST(pivoting);

  if (mode != mode_full && mode && mode_r &&
    mode != mode_economic && mode != mode_raw) {
    //raise ArgumentError: invalid mode
  }

  nmatrix* result_qr = ALLOC(nmatrix);
  result_qr->dtype = matrix->dtype;
  result_qr->stype = matrix->stype;
  result_qr->ndims = matrix->ndims;
  result_qr->shape = ALLOC_N(size_t, result_qr->ndims);

  result_qr->shape[0] =  matrix->shape[0];
  result_qr->shape[1] =  matrix->shape[1];
  result_qr->count = matrix->count;

  nmatrix* result_tau = ALLOC(nmatrix);
  result_tau->dtype = matrix->dtype;
  result_tau->stype = matrix->stype;
  result_tau->ndims = matrix->ndims;
  result_tau->shape = ALLOC_N(size_t, result_tau->ndims);

  result_tau->shape[0] =  matrix->shape[0];
  result_tau->shape[1] =  matrix->shape[1];
  result_tau->count = matrix->count;

  nmatrix* result_q = ALLOC(nmatrix);
  result_q->dtype = matrix->dtype;
  result_q->stype = matrix->stype;
  result_q->ndims = matrix->ndims;
  result_q->shape = ALLOC_N(size_t, result_q->ndims);

  result_q->shape[0] =  matrix->shape[0];
  result_q->shape[1] =  matrix->shape[1];
  result_q->count = matrix->count;

  nmatrix* result_r = ALLOC(nmatrix);
  result_r->dtype = matrix->dtype;
  result_r->stype = matrix->stype;
  result_r->ndims = matrix->ndims;
  result_r->shape = ALLOC_N(size_t, result_r->ndims);

  result_r->shape[0] =  matrix->shape[0];
  result_r->shape[1] =  matrix->shape[1];
  result_r->count = matrix->count;

  if(pivoting == true) {
    switch(matrix->dtype) {
      case nm_bool:
      {
        //raise not supported error
      }
      case nm_int:
      {
        //raise not supported error
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
  }
  else {
    switch(matrix->dtype) {
      case nm_bool:
      {
        //raise not supported error
      }
      case nm_int:
      {
        //raise not supported error
      }
      case nm_float32:
      {
        int m = matrix->shape[0]; //no. of rows
        int n = matrix->shape[1]; //no. of cols
        int lda = m;
        int info = -1;
        float* tau = ALLOC_N(float, min(m, n));
        float* elements = ALLOC_N(float, result->count);

        memcpy(elements, matrix->elements, sizeof(float)*result->count);

        info = LAPACKE_sgeqrf(LAPACK_ROW_MAJOR, m, n, a, lda, tau);

        result_qr->elements = elements;
        result_tau->elements = tau;

        if(mode == mode_raw) {
          VALUE ary = rb_ary_new();
          rb_ary_push(ary, Data_Wrap_Struct(NMatrix, NULL, nm_free, result_qr));
          rb_ary_push(ary, Data_Wrap_Struct(NMatrix, NULL, nm_free, result_tau));

          return ary;
          //return [qr, tou]
        }
        else if(mode == mode_r) {
          //TODO
        }
        else if(mode == mode_economic) {
          //TODO
        }
        else if(mode == mode_full) {
          //TODO
        }
      }
      case nm_float64:
      {
        //lapack_int LAPACKE_dgeqrf( int matrix_layout, lapack_int m, lapack_int n,
        //                   double* a, lapack_int lda, double* tau );
        int m = matrix->shape[0]; //no. of rows
        int n = matrix->shape[1]; //no. of cols
        int lda = m;
        int info = -1;
        double* tau = ALLOC_N(double, min(m, n));
        double* elements = ALLOC_N(double, result->count);

        memcpy(elements, matrix->elements, sizeof(double)*result->count);

        info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, a, lda, tau);

        result_qr->elements = elements;
        result_tau->elements = tau;

        if(mode == mode_raw) {
          VALUE ary = rb_ary_new();
          rb_ary_push(ary, Data_Wrap_Struct(NMatrix, NULL, nm_free, result_qr));
          rb_ary_push(ary, Data_Wrap_Struct(NMatrix, NULL, nm_free, result_tau));

          return ary;
          //return [qr, tou]
        }
        else if(mode == mode_r) {
          //TODO
        }
        else if(mode == mode_economic) {
          //TODO
        }
        else if(mode == mode_full) {
          //TODO
        }
      }
      case nm_complex32:
      {
        int m = matrix->shape[0]; //no. of rows
        int n = matrix->shape[1]; //no. of cols
        int lda = m;
        int info = -1;
        float complex* tau = ALLOC_N(float complex, min(m, n));
        float complex* elements = ALLOC_N(float complex, result->count);

        memcpy(elements, matrix->elements, sizeof(float complex)*result->count);

        info = LAPACKE_cgeqrf(LAPACK_ROW_MAJOR, m, n, a, lda, tau);

        result_qr->elements = elements;
        result_tau->elements = tau;

        if(mode == mode_raw) {
          VALUE ary = rb_ary_new();
          rb_ary_push(ary, Data_Wrap_Struct(NMatrix, NULL, nm_free, result_qr));
          rb_ary_push(ary, Data_Wrap_Struct(NMatrix, NULL, nm_free, result_tau));

          return ary;
          //return [qr, tou]
        }
        else if(mode == mode_r) {
          //TODO
        }
        else if(mode == mode_economic) {
          //TODO
        }
        else if(mode == mode_full) {
          //TODO
        }
      }
      case nm_complex64:
      {
        int m = matrix->shape[0]; //no. of rows
        int n = matrix->shape[1]; //no. of cols
        int lda = m;
        int info = -1;
        double complex* tau = ALLOC_N(double complex, min(m, n));
        double complex* elements = ALLOC_N(double complex, result->count);

        memcpy(elements, matrix->elements, sizeof(double complex)*result->count);

        info = LAPACKE_zgeqrf(LAPACK_ROW_MAJOR, m, n, a, lda, tau);

        result_qr->elements = elements;
        result_tau->elements = tau;

        if(mode == mode_raw) {
          VALUE ary = rb_ary_new();
          rb_ary_push(ary, Data_Wrap_Struct(NMatrix, NULL, nm_free, result_qr));
          rb_ary_push(ary, Data_Wrap_Struct(NMatrix, NULL, nm_free, result_tau));

          return ary;
          //return [qr, tou]
        }
        else if(mode == mode_r) {
          //TODO
        }
        else if(mode == mode_economic) {
          //TODO
        }
        else if(mode == mode_full) {
          //TODO
        }
      }
    }
  }


}
