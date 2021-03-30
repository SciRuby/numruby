/*
 *	Calculates the dot product of two matrices.
 *	Args:
 *	-	self matrix, type: NMatrix
 *	-	another matrix, type: NMatrix
 *	
 *	returns the resultant matrix of type NMatrix
 */
VALUE nm_dot(VALUE self, VALUE another){
  nmatrix* left;
  nmatrix* right;
  TypedData_Get_Struct(self, nmatrix, &nm_data_type, left);
  TypedData_Get_Struct(another, nmatrix, &nm_data_type, right);

  nmatrix* result = ALLOC(nmatrix);
  result->dtype = left->dtype;
  result->stype = left->stype;
  result->ndims = left->ndims;
  result->shape = ALLOC_N(size_t, result->ndims);

  result->shape[0] =  left->shape[0];
  result->shape[1] = right->shape[1];
  result->count = result->shape[0] * result->shape[1];

  switch (left->dtype) {
    case nm_bool:
    {
      // Not supported message and casting to double
      break;
    }
    case nm_int:
    {
      // Not supported message and casting to double
      break;
    }
    case nm_float64:
    {
      result->elements = ALLOC_N(double, result->shape[0] * result->shape[1]);
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)left->shape[0], (int)right->shape[1], (int)left->shape[1], /*no scaling*/
                  1, left->elements, (int)left->shape[1], right->elements, (int)right->shape[1], /*no addition*/0, result->elements, (int)right->shape[1]);
      break;
    }
    case nm_float32:
    {
      result->elements = ALLOC_N(float, result->shape[0] * result->shape[1]);
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)left->shape[0], (int)right->shape[1], (int)left->shape[1], /*no scaling*/
                  1, left->elements, (int)left->shape[1], right->elements, (int)right->shape[1], /*no addition*/0, result->elements, (int)right->shape[1]);
      break;
    }
    case nm_complex32:
    {
      float alpha[2] = {1, 1};
      float beta[2]  = {0, 0};
      result->elements = ALLOC_N(complex float, result->shape[0] * result->shape[1]);
      cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)left->shape[0], (int)right->shape[1], (int)left->shape[1], /*no scaling*/
                  alpha, left->elements, (int)left->shape[1], right->elements, (int)right->shape[1], /*no addition*/beta, result->elements, (int)right->shape[1]);
      break;
    }
    case nm_complex64:
    {
      double alpha[2] = {1, 1};
      double beta[2]  = {0, 0};
      result->elements = ALLOC_N(complex double, result->shape[0] * result->shape[1]);
      cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)left->shape[0], (int)right->shape[1], (int)left->shape[1], /*no scaling*/
                  alpha, left->elements, (int)left->shape[1], right->elements, (int)right->shape[1], /*no addition*/beta, result->elements, (int)right->shape[1]);
      break;
    }
  }

  return TypedData_Wrap_Struct(NMatrix, &nm_data_type, result);
}

/*
 *	Calculates matrix norm.
 *	Args:
 *	-	self matrix, type: NMatrix
 *	
 *	returns the norm matrix of type float
 */
VALUE nm_norm2(VALUE self){
  nmatrix* matrix;
  TypedData_Get_Struct(self, nmatrix, &nm_data_type, matrix);
  //check mat is vector
  VALUE val = Qnil;

  switch (matrix->dtype) {
    case nm_bool:
    {
      // Not supported message and casting to double
      break;
    }
    case nm_int:
    {
      // Not supported message and casting to double
      break;
    }
    case nm_float32:
    {
      float norm = cblas_snrm2(matrix->count, matrix->elements, 1);
      val = DBL2NUM(norm);
      break;
    }
    case nm_float64:
    {
      double norm = cblas_dnrm2(matrix->count, matrix->elements, 1);
      val = DBL2NUM(norm);
      break;
    }
    case nm_complex32:
    {
      double norm = cblas_dznrm2(matrix->count, matrix->elements, 1);
      val = DBL2NUM(norm);
      break;
    }
    case nm_complex64:
    {
      double norm = cblas_dznrm2(matrix->count, matrix->elements, 1);
      val = DBL2NUM(norm);
      break;
    }
  }

  return val;
}
