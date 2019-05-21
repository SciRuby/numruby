/*
 * Calculates the weighted average of elements across a given axis (dimension).
 * Returns a list of weighted averages having length equal to the length of given axis.
 *
 * Args:
 * 	- input matrix, type: NMatrix
 *	- axis (dimension), type: integer
 *	- weight matrix, type: NMatrix
 */
VALUE average_nmatrix(int argc, VALUE* argv){
  nmatrix* matrix;
  Data_Get_Struct(argv[0], nmatrix, matrix);

  size_t axis = NUM2LONG(argv[1]);

  nmatrix* weights;
  Data_Get_Struct(argv[2], nmatrix, weights);
  double* weight_elements = weights->elements;

  nmatrix* result = ALLOC(nmatrix);
  result->dtype = matrix->dtype;
  result->stype = matrix->stype;
  result->ndims = matrix->ndims;
  result->shape = ALLOC_N(size_t, result->ndims);

  result->shape[1] = 1;


  if(axis == 0){
    result->shape[0] = matrix->shape[0];
    result->count = result->shape[0] * result->shape[1];
    double* matrix_elements = (double*)matrix->elements;
    double* result_elements = ALLOC_N(double, matrix->shape[0]);
    for(size_t i = 0; i < matrix->shape[0]; ++i){
      double sum = 0;
      for(size_t j = 0; j < matrix->shape[1]; ++j){
        sum += matrix_elements[i*matrix->shape[1] + j] * weight_elements[j];
      }
      result_elements[i] = sum / matrix->shape[0];
    }
    result->elements = result_elements;
  }else{
    result->shape[0] = matrix->shape[1];
    result->count = result->shape[0] * result->shape[1];
    double* matrix_elements = (double*)matrix->elements;
    double* result_elements = ALLOC_N(double, matrix->shape[1]);
    for(size_t i = 0; i < matrix->shape[1]; ++i){
      double sum = 0;
      for(size_t j = 0; j < matrix->shape[0]; ++j){
        sum += matrix_elements[j*matrix->shape[1] + i] * weight_elements[j];
      }
      result_elements[i] = sum / matrix->shape[1];
    }
    result->elements = result_elements;
  }

  return Data_Wrap_Struct(NMatrix, NULL, nm_free, result);
}
