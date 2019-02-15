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
  getrf(matrix->elements, matrix->shape[1], matrix->shape[0], ipiv, elements);
  int lda = n;

  LAPACKE_dgetri(101, n, elements, lda, ipiv);
  result->elements = elements;
  return Data_Wrap_Struct(NMatrix, NULL, nm_free, result);
}


void getrf(const double* arr, const size_t cols, const size_t rows, int* ipiv, double* arr2) {
  int m = (int)cols;
  int n = (int)rows;
  memcpy(arr2, arr, sizeof(double)*rows*cols);
  int lda = m;
  LAPACKE_dgetrf(101,m,n,arr2,lda,ipiv);
}

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

VALUE nm_det(VALUE self){
  nmatrix* matrix;
  Data_Get_Struct(self, nmatrix, matrix);

  int n = (int)matrix->shape[1];
  int m = (int)matrix->shape[0];

  double* elements = ALLOC_N(double, matrix->count);
  int* pivot = ALLOC_N(int, min(m,n)+1);

  getrf(matrix->elements, matrix->shape[1], matrix->shape[0], pivot, elements);

  int num_perm = 0;
  int j = 0;
  for(int i = 0; i < min(m,n)+1; ++i){
    if(pivot[i]-1 != j){num_perm += 1;}
    j++;
  }
  double prod;

  prod = (num_perm % 2 == 1) ? 1 : -1;

  for(int i =0; i < min(m,n); i++){
    prod *= elements[matrix->shape[0]*i + i];
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
  return Qnil;
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

VALUE nm_qr(VALUE self){
  return Qnil;
}
