VALUE nm_dot(VALUE self, VALUE another){
  nmatrix* left;
  nmatrix* right;
  Data_Get_Struct(self, nmatrix, left);
  Data_Get_Struct(another, nmatrix, right);

  nmatrix* result = ALLOC(nmatrix);
  result->count = left->count;
  result->ndims = left->ndims;
  result->shape = ALLOC_N(size_t, result->ndims);

  result->shape[0] =  left->shape[0];
  result->shape[1] = right->shape[1];

  result->elements = ALLOC_N(double, result->shape[0] * result->shape[1]);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)left->shape[0], (int)right->shape[1], (int)left->shape[1], /*no scaling*/
       1, left->elements, (int)left->shape[1], right->elements, (int)right->shape[1], /*no addition*/0, result->elements, (int)right->shape[1]);
  return Data_Wrap_Struct(NMatrix, NULL, nm_free, result);
}
