/*
 * Equality operator. Returns a single true or false value indicating whether
 * the matrices are equivalent.
 *
 */
VALUE nm_eqeq(VALUE self, VALUE another){
  nmatrix* left;
  nmatrix* right;
  Data_Get_Struct(self, nmatrix, left);
  Data_Get_Struct(another, nmatrix, right);

  if(left->count != right->count){
    return Qfalse;
  }

  switch(right->dtype) {
    case nm_bool: 
    {
      bool* left_elements = (bool*)left->elements;
      bool* right_elements = (bool*)right->elements;

      for(size_t index = 0; index < left->count; index++){
        if(left_elements[index] != right_elements[index]){
          return Qfalse;
        }
      }
      break;
    }
    case nm_int: 
    {
      int* left_elements = (int*)left->elements;
      int* right_elements = (int*)right->elements;

      for(size_t index = 0; index < left->count; index++){
        if(left_elements[index] != right_elements[index]){
          return Qfalse;
        }
      }
      break;
    }
    case nm_float32: 
    {
      float* left_elements = (float*)left->elements;
      float* right_elements = (float*)right->elements;

      for(size_t index = 0; index < left->count; index++){
        if(fabs(left_elements[index] - right_elements[index]) > 1e-3){
          return Qfalse;
        }
      }
      break;
    }
    case nm_float64: 
    {
      double* left_elements = (double*)left->elements;
      double* right_elements = (double*)right->elements;

      for(size_t index = 0; index < left->count; index++){
        if(fabs(left_elements[index] - right_elements[index]) > 1e-3){
          return Qfalse;
        }
      }
      break;
    }
    case nm_complex32: 
    {
      float complex* left_elements = (float complex*)left->elements;
      float complex* right_elements = (float complex*)right->elements;

      for(size_t index = 0; index < left->count; index++){
        if(cabs(left_elements[index] - right_elements[index]) > 1e-3){
          return Qfalse;
        }
      }
      break;
    }
    case nm_complex64: 
    {
      double complex* left_elements = (double complex*)left->elements;
      double complex* right_elements = (double complex*)right->elements;

      for(size_t index = 0; index < left->count; index++){
        if(cabs(left_elements[index] - right_elements[index]) > 1e-3){
          return Qfalse;
        }
      }
      break;
    }
  }

  return Qtrue;
}

/*
 * Greater operator.
 * Returns a single true or false value indicating whether
 * the element in a matrix is greater or smaller than given value
 *
 */
VALUE nm_gt(VALUE self, VALUE another){
  nmatrix* left;
  Data_Get_Struct(self, nmatrix, left);

  nmatrix* result = ALLOC(nmatrix);
  result->dtype = nm_bool;
  result->stype = left->stype;
  result->count = left->count;
  result->ndims = left->ndims;
  result->shape = ALLOC_N(size_t, result->ndims);

  result->shape[0] = left->shape[0];
  result->shape[1] = left->shape[1];

  bool* result_elements = ALLOC_N(bool, result->shape[0] * result->shape[1]);

  switch (left->dtype) {
    case nm_bool: {
      double rha = NUM2DBL(another);
      bool* left_elements = (bool*)left->elements;

      for(size_t index = 0; index < left->count; index++){
        result_elements[index] = (((double)left_elements[index] > rha) ? true : false);
      }
      break;
    }
    case nm_int: {
      double rha = NUM2DBL(another);
      int* left_elements = (int*)left->elements;

      for(size_t index = 0; index < left->count; index++){
        result_elements[index] = (((double)left_elements[index] > rha) ? true : false);
      }
      break;
    }
    case nm_float32: {
      double rha = NUM2DBL(another);
      float* left_elements = (float*)left->elements;

      for(size_t index = 0; index < left->count; index++){
        result_elements[index] = (((double)left_elements[index] > rha) ? true : false);
      }
      break;
    }
    case nm_float64: {
      double rha = NUM2DBL(another);
      double* left_elements = (double*)left->elements;

      for(size_t index = 0; index < left->count; index++){
        result_elements[index] = ((left_elements[index] > rha) ? true : false);
      }
      break;
    }
    case nm_complex32: {
      rb_raise(rb_eSyntaxError, "SyntaxError: nm_complex32 does not support this operator.");
      // float complex rha = CMPLXF(NUM2DBL(rb_funcall(another, rb_intern("real"), 0, Qnil)), NUM2DBL(rb_funcall(another, rb_intern("imaginary"), 0, Qnil)));;
      // float complex* left_elements = (float complex*)left->elements;

      // for(size_t index = 0; index < left->count; index++){
      //   result_elements[index] = ((left_elements[index] > rha) ? true : false);
      // }
      break;
    }
    case nm_complex64: {
      rb_raise(rb_eSyntaxError, "SyntaxError: nm_complex64 does not support this operator.");
      // double complex rha = CMPLXF(NUM2DBL(rb_funcall(another, rb_intern("real"), 0, Qnil)), NUM2DBL(rb_funcall(another, rb_intern("imaginary"), 0, Qnil)));
      // double complex* left_elements = (double complex*)left->elements;

      // for(size_t index = 0; index < left->count; index++){
      //   result_elements[index] = ((left_elements[index] > rha) ? true : false);
      // }
      break;
    }
  }
  result->elements = result_elements;
  return Data_Wrap_Struct(NMatrix, NULL, nm_free, result);
}

/*
 * Greater operator.
 * Returns a single true or false value indicating whether
 * the element in a matrix is greater or smaller
 *
 */
VALUE nm_gteq(VALUE self, VALUE another){
  nmatrix* left;
  Data_Get_Struct(self, nmatrix, left);

  nmatrix* result = ALLOC(nmatrix);
  result->dtype = nm_bool;
  result->stype = left->stype;
  result->count = left->count;
  result->ndims = left->ndims;
  result->shape = ALLOC_N(size_t, result->ndims);

  result->shape[0] = left->shape[0];
  result->shape[1] = left->shape[1];

  bool* result_elements = ALLOC_N(bool, result->shape[0] * result->shape[1]);

  switch (left->dtype) {
    case nm_bool: {
      double rha = NUM2DBL(another);
      bool* left_elements = (bool*)left->elements;

      for(size_t index = 0; index < left->count; index++){
        result_elements[index] = (((double)left_elements[index] >= rha) ? true : false);
      }
      break;
    }
    case nm_int: {
      double rha = NUM2DBL(another);
      int* left_elements = (int*)left->elements;

      for(size_t index = 0; index < left->count; index++){
        result_elements[index] = (((double)left_elements[index] >= rha) ? true : false);
      }
      break;
    }
    case nm_float32: {
      double rha = NUM2DBL(another);
      float* left_elements = (float*)left->elements;

      for(size_t index = 0; index < left->count; index++){
        result_elements[index] = (((double)left_elements[index] >= rha) ? true : false);
      }
      break;
    }
    case nm_float64: {
      double rha = NUM2DBL(another);
      double* left_elements = (double*)left->elements;

      for(size_t index = 0; index < left->count; index++){
        result_elements[index] = ((left_elements[index] >= rha) ? true : false);
      }
      break;
    }
    case nm_complex32: {
      rb_raise(rb_eSyntaxError, "SyntaxError: nm_complex32 does not support this operator.");
      // float complex rha = CMPLXF(NUM2DBL(rb_funcall(another, rb_intern("real"), 0, Qnil)), NUM2DBL(rb_funcall(another, rb_intern("imaginary"), 0, Qnil)));;
      // float complex* left_elements = (float complex*)left->elements;

      // for(size_t index = 0; index < left->count; index++){
      //   result_elements[index] = ((left_elements[index] >= rha) ? true : false);
      // }
      break;
    }
    case nm_complex64: {
      rb_raise(rb_eSyntaxError, "SyntaxError: nm_complex64 does not support this operator.");
      // double complex rha = CMPLXF(NUM2DBL(rb_funcall(another, rb_intern("real"), 0, Qnil)), NUM2DBL(rb_funcall(another, rb_intern("imaginary"), 0, Qnil)));
      // double complex* left_elements = (double complex*)left->elements;

      // for(size_t index = 0; index < left->count; index++){
      //   result_elements[index] = ((left_elements[index] >= rha) ? true : false);
      // }
      break;
    }
  }
  result->elements = result_elements;
  return Data_Wrap_Struct(NMatrix, NULL, nm_free, result);
}

/*
 * Greater operator.
 * Returns a single true or false value indicating whether
 * the element in a matrix is greater or smaller
 *
 */
VALUE nm_lt(VALUE self, VALUE another){
  nmatrix* left;
  Data_Get_Struct(self, nmatrix, left);

  nmatrix* result = ALLOC(nmatrix);
  result->dtype = nm_bool;
  result->stype = left->stype;
  result->count = left->count;
  result->ndims = left->ndims;
  result->shape = ALLOC_N(size_t, result->ndims);

  result->shape[0] = left->shape[0];
  result->shape[1] = left->shape[1];

  bool* result_elements = ALLOC_N(bool, result->shape[0] * result->shape[1]);

  switch (left->dtype) {
    case nm_bool: {
      double rha = NUM2DBL(another);
      bool* left_elements = (bool*)left->elements;

      for(size_t index = 0; index < left->count; index++){
        result_elements[index] = (((double)left_elements[index] < rha) ? true : false);
      }
      break;
    }
    case nm_int: {
      double rha = NUM2DBL(another);
      int* left_elements = (int*)left->elements;

      for(size_t index = 0; index < left->count; index++){
        result_elements[index] = (((double)left_elements[index] < rha) ? true : false);
      }
      break;
    }
    case nm_float32: {
      double rha = NUM2DBL(another);
      float* left_elements = (float*)left->elements;

      for(size_t index = 0; index < left->count; index++){
        result_elements[index] = (((double)left_elements[index] < rha) ? true : false);
      }
      break;
    }
    case nm_float64: {
      double rha = NUM2DBL(another);
      double* left_elements = (double*)left->elements;

      for(size_t index = 0; index < left->count; index++){
        result_elements[index] = ((left_elements[index] < rha) ? true : false);
      }
      break;
    }
    case nm_complex32: {
      rb_raise(rb_eSyntaxError, "SyntaxError: nm_complex32 does not support this operator.");
      // float complex rha = CMPLXF(NUM2DBL(rb_funcall(another, rb_intern("real"), 0, Qnil)), NUM2DBL(rb_funcall(another, rb_intern("imaginary"), 0, Qnil)));;
      // float complex* left_elements = (float complex*)left->elements;

      // for(size_t index = 0; index < left->count; index++){
      //   result_elements[index] = ((left_elements[index] < rha) ? true : false);
      // }
      break;
    }
    case nm_complex64: {
      rb_raise(rb_eSyntaxError, "SyntaxError: nm_complex64 does not support this operator.");
      // double complex rha = CMPLXF(NUM2DBL(rb_funcall(another, rb_intern("real"), 0, Qnil)), NUM2DBL(rb_funcall(another, rb_intern("imaginary"), 0, Qnil)));
      // double complex* left_elements = (double complex*)left->elements;

      // for(size_t index = 0; index < left->count; index++){
      //   result_elements[index] = ((left_elements[index] < rha) ? true : false);
      // }
      break;
    }
  }
  result->elements = result_elements;
  return Data_Wrap_Struct(NMatrix, NULL, nm_free, result);
}


/*
 * Greater operator.
 * Returns a single true or false value indicating whether
 * the element in a matrix is greater or smaller
 *
 */
VALUE nm_lteq(VALUE self, VALUE another){
  nmatrix* left;
  Data_Get_Struct(self, nmatrix, left);

  nmatrix* result = ALLOC(nmatrix);
  result->dtype = nm_bool;
  result->stype = left->stype;
  result->count = left->count;
  result->ndims = left->ndims;
  result->shape = ALLOC_N(size_t, result->ndims);

  result->shape[0] = left->shape[0];
  result->shape[1] = left->shape[1];

  bool* result_elements = ALLOC_N(bool, result->shape[0] * result->shape[1]);

  switch (left->dtype) {
    case nm_bool: {
      double rha = NUM2DBL(another);
      bool* left_elements = (bool*)left->elements;

      for(size_t index = 0; index < left->count; index++){
        result_elements[index] = (((double)left_elements[index] <= rha) ? true : false);
      }
      break;
    }
    case nm_int: {
      double rha = NUM2DBL(another);
      int* left_elements = (int*)left->elements;

      for(size_t index = 0; index < left->count; index++){
        result_elements[index] = (((double)left_elements[index] <= rha) ? true : false);
      }
      break;
    }
    case nm_float32: {
      double rha = NUM2DBL(another);
      float* left_elements = (float*)left->elements;

      for(size_t index = 0; index < left->count; index++){
        result_elements[index] = (((double)left_elements[index] <= rha) ? true : false);
      }
      break;
    }
    case nm_float64: {
      double rha = NUM2DBL(another);
      double* left_elements = (double*)left->elements;

      for(size_t index = 0; index < left->count; index++){
        result_elements[index] = ((left_elements[index] <= rha) ? true : false);
      }
      break;
    }
    case nm_complex32: {
      rb_raise(rb_eSyntaxError, "SyntaxError: nm_complex32 does not support this operator.");
      // float complex rha = CMPLXF(NUM2DBL(rb_funcall(another, rb_intern("real"), 0, Qnil)), NUM2DBL(rb_funcall(another, rb_intern("imaginary"), 0, Qnil)));;
      // float complex* left_elements = (float complex*)left->elements;

      // for(size_t index = 0; index < left->count; index++){
      //   result_elements[index] = ((left_elements[index] <= rha) ? true : false);
      // }
      break;
    }
    case nm_complex64: {
      rb_raise(rb_eSyntaxError, "SyntaxError: nm_complex64 does not support this operator.");
      // double complex rha = CMPLXF(NUM2DBL(rb_funcall(another, rb_intern("real"), 0, Qnil)), NUM2DBL(rb_funcall(another, rb_intern("imaginary"), 0, Qnil)));
      // double complex* left_elements = (double complex*)left->elements;

      // for(size_t index = 0; index < left->count; index++){
      //   result_elements[index] = ((left_elements[index] <= rha) ? true : false);
      // }
      break;
    }
  }
  result->elements = result_elements;
  return Data_Wrap_Struct(NMatrix, NULL, nm_free, result);
}