/*
 * Elementiwise operator. Returns a matrix which is the elementwise operation
 * of the two operands (left one is always a matrix, right one could be 
 * a matrix of a single value).
 *
 */
#define DEF_ELEMENTWISE_RUBY_ACCESSOR(name, oper)  \
VALUE nm_##name(VALUE self, VALUE another){        \
  nmatrix* left;                                   \
  Data_Get_Struct(self, nmatrix, left);            \
                                                   \
  nmatrix* right;                                  \
  nmatrix* result = ALLOC(nmatrix);                \
                                                   \
  nmatrix* left_copy;                              \
  nmatrix* right_copy;                             \
  if(rb_obj_is_kind_of(another, NMatrix) == Qtrue) {\
    Data_Get_Struct(another, nmatrix, right);       \
    left_copy = matrix_copy(left);                  \
    right_copy = matrix_copy(right);                \
    broadcast_matrices(left_copy, right_copy);      \
    result->dtype = left_copy->dtype;                 \
    result->stype = left_copy->stype;                 \
    result->count = left_copy->count;                 \
    result->ndims = left_copy->ndims;                 \
    result->shape = ALLOC_N(size_t, result->ndims);  \
                                                    \
    for(size_t index = 0; index < result->ndims; index++){\
      result->shape[index] = left_copy->shape[index];     \
    }                                                     \
  }                                                       \
  else {                                         \
    result->dtype = left->dtype;                 \
    result->stype = left->stype;                 \
    result->count = left->count;                 \
    result->ndims = left->ndims;                 \
    result->shape = ALLOC_N(size_t, result->ndims);   \
                                                      \
    for(size_t index = 0; index < result->ndims; index++){\
      result->shape[index] = left->shape[index];          \
    }                                                     \
  }                                                       \
  switch (result->dtype) {                                                       \
    case nm_bool:                                                                 \
    {                                                                            \
      bool* result_elements = ALLOC_N(bool, result->count);                  \
      if(RB_TYPE_P(another, T_TRUE) || RB_TYPE_P(another, T_FALSE)){           \
        bool* left_elements = (bool*)left->elements;                       \
        for(size_t index = 0; index < left->count; index++){                     \
          result_elements[index] = (left_elements[index]) oper (another ? Qtrue : Qfalse);      \
        }                                                                        \
      }                                                                          \
      else{                                                                      \
        bool* left_elements = (bool*)left_copy->elements;                        \
        bool* right_elements = (bool*)right_copy->elements;                           \
                                                                                 \
        for(size_t index = 0; index < left_copy->count; index++){                     \
          result_elements[index] = (left_elements[index]) oper (right_elements[index]); \
        }                                                                        \
      }                                                                          \
      result->elements = result_elements;                                        \
      break;                                                                     \
    }                                                                            \
    case nm_int:                                                                 \
    {                                                                            \
      int* result_elements = ALLOC_N(int, result->count);                  \
      if(RB_TYPE_P(another, T_FLOAT) || RB_TYPE_P(another, T_FIXNUM)){           \
        int* left_elements = (int*)left->elements;                               \
        for(size_t index = 0; index < left->count; index++){                     \
          result_elements[index] = (left_elements[index]) oper (NUM2DBL(another));      \
        }                                                                        \
      }                                                                          \
      else{                                                                      \
        int* left_elements = (int*)left_copy->elements;                          \
        int* right_elements = (int*)right_copy->elements;                             \
                                                                                 \
        for(size_t index = 0; index < left_copy->count; index++){                     \
          result_elements[index] = (left_elements[index]) oper (right_elements[index]); \
        }                                                                        \
      }                                                                          \
      result->elements = result_elements;                                        \
      break;                                                                     \
    }                                                                            \
    case nm_float32:                                                             \
    {                                                                            \
      float* result_elements = ALLOC_N(float, result->count);                    \
      if(RB_TYPE_P(another, T_FLOAT) || RB_TYPE_P(another, T_FIXNUM)){           \
        float* left_elements = (float*)left->elements;                           \
        for(size_t index = 0; index < left->count; index++){                     \
          result_elements[index] = (left_elements[index]) oper (NUM2DBL(another));      \
        }                                                                        \
      }                                                                          \
      else{                                                                      \
        float* left_elements = (float*)left_copy->elements;                      \
        float* right_elements = (float*)right_copy->elements;                         \
                                                                                 \
        for(size_t index = 0; index < left_copy->count; index++){                       \
          result_elements[index] = (left_elements[index]) oper (right_elements[index]);   \
        }                                                                          \
      }                                                                            \
      result->elements = result_elements;                                          \
      break;                                                                       \
    }                                                                             \
    case nm_float64:                                                             \
    {                                                                            \
      double* result_elements = ALLOC_N(double, result->count);                  \
      if(RB_TYPE_P(another, T_FLOAT) || RB_TYPE_P(another, T_FIXNUM)){           \
        double* left_elements = (double*)left->elements;                    \
        for(size_t index = 0; index < left->count; index++){                     \
          result_elements[index] = (left_elements[index]) oper (NUM2DBL(another));      \
        }                                                                        \
      }                                                                          \
      else{                                                                      \
        double* left_elements = (double*)left_copy->elements;                    \
        double* right_elements = (double*)right_copy->elements;                       \
                                                                                 \
        for(size_t index = 0; index < left_copy->count; index++){                     \
          result_elements[index] = (left_elements[index]) oper (right_elements[index]); \
        }                                                                        \
      }                                                                          \
      result->elements = result_elements;                                        \
      break;                                                                     \
    }                                                                            \
    case nm_complex32:                                                             \
    {                                                                            \
      complex float* result_elements = ALLOC_N(complex float, result->count);                    \
      if(RB_TYPE_P(another, T_FLOAT) || RB_TYPE_P(another, T_FIXNUM)){           \
        complex float* left_elements = (complex float*)left->elements;      \
        for(size_t index = 0; index < left->count; index++){                     \
          result_elements[index] = (left_elements[index]) oper (NUM2DBL(another));      \
        }                                                                        \
      }                                                                          \
      else{                                                                      \
        complex float* left_elements = (complex float*)left_copy->elements;      \
        complex float* right_elements = (complex float*)right_copy->elements;         \
                                                                                 \
        for(size_t index = 0; index < left_copy->count; index++){                       \
          result_elements[index] = (left_elements[index]) oper (right_elements[index]);   \
        }                                                                          \
      }                                                                            \
      result->elements = result_elements;                                          \
      break;                                                                       \
    }                                                                              \
    case nm_complex64:                                                             \
    {                                                                            \
      complex double* result_elements = ALLOC_N(complex double, result->count);                    \
      if(RB_TYPE_P(another, T_FLOAT) || RB_TYPE_P(another, T_FIXNUM)){           \
        complex double* left_elements = (complex double*)left->elements;    \
        for(size_t index = 0; index < left->count; index++){                     \
          result_elements[index] = (left_elements[index]) oper (NUM2DBL(another));      \
        }                                                                        \
      }                                                                          \
      else{                                                                      \
        complex double* left_elements = (complex double*)left_copy->elements;    \
        complex double* right_elements = (complex double*)right_copy->elements;       \
                                                                                 \
        for(size_t index = 0; index < left_copy->count; index++){                       \
          result_elements[index] = (left_elements[index]) oper (right_elements[index]);   \
        }                                                                          \
      }                                                                            \
      result->elements = result_elements;                                          \
      break;                                                                       \
    }                                                                              \
  }                                                                                 \
  return Data_Wrap_Struct(NMatrix, NULL, nm_free, result);                         \
}

DEF_ELEMENTWISE_RUBY_ACCESSOR(add, +)
DEF_ELEMENTWISE_RUBY_ACCESSOR(subtract, -)
DEF_ELEMENTWISE_RUBY_ACCESSOR(multiply, *)
DEF_ELEMENTWISE_RUBY_ACCESSOR(divide, /)
//DEF_ELEMENTWISE_RUBY_ACCESSOR(divide, ^) this should be for exponentiation (or use **)

/*
 *  Elementwise sin operator.
 *  Takes in the given matrix
 *  and returns the matrix with each element
 *  as corresponding sin of the value of that element.
*/

VALUE nm_sin(VALUE self){
  nmatrix* input;
  Data_Get_Struct(self, nmatrix, input);

  nmatrix* result = ALLOC(nmatrix);
  result->dtype = input->dtype;
  result->stype = input->stype;
  result->count = input->count;
  result->ndims = input->ndims;
  result->shape = ALLOC_N(size_t, result->ndims);

  for(size_t index = 0; index < result->ndims; index++){
    result->shape[index] = input->shape[index];
  }

  switch(result->dtype){
    case nm_bool:
    {
      bool* input_elements = (bool*)input->elements;
      bool* result_elements = ALLOC_N(bool, result->count);
      for(size_t index = 0; index < input->count; index++){
        result_elements[index] = sin(input_elements[index]);
      }
      result->elements = result_elements;
      break;
    }
    case nm_int:
    {
      int* input_elements = (int*)input->elements;
      int* result_elements = ALLOC_N(int, result->count);
      for(size_t index = 0; index < input->count; index++){
        result_elements[index] = sin(input_elements[index]);
      }
      result->elements = result_elements;
      break;
    }
    case nm_float32:
    {
      float* input_elements = (float*)input->elements;
      float* result_elements = ALLOC_N(float, result->count);
      for(size_t index = 0; index < input->count; index++){
        result_elements[index] = sin(input_elements[index]);
      }
      result->elements = result_elements;
      break;
    }
    case nm_float64:
    {
      double* input_elements = (double*)input->elements;
      double* result_elements = ALLOC_N(double, result->count);
      for(size_t index = 0; index < input->count; index++){
        result_elements[index] = sin(input_elements[index]);
      }
      result->elements = result_elements;
      break;
    }
    case nm_complex32:
    {
      complex float* input_elements = (complex float*)input->elements;
      complex float* result_elements = ALLOC_N(complex float, result->count);
      for(size_t index = 0; index < input->count; index++){
        result_elements[index] = csin(input_elements[index]);
      }
      result->elements = result_elements;
      break;
    }
    case nm_complex64:
    {
      complex double* input_elements = (complex double*)input->elements;
      complex double* result_elements = ALLOC_N(complex double, result->count);
      for(size_t index = 0; index < input->count; index++){
        result_elements[index] = csin(input_elements[index]);
      }
      result->elements = result_elements;
      break;
    }
  }

  return Data_Wrap_Struct(NMatrix, NULL, nm_free, result);
}

/*
 *  Elementwise sum operator.
 *  Takes in the given matrix
 *  and returns the sum of all the elements of the matrix
*/
VALUE nm_sum(VALUE self){
  nmatrix* input;
  Data_Get_Struct(self, nmatrix, input);

  VALUE result;


  switch(input->dtype){
    case nm_bool:
    {
      bool* input_elements = (bool*)input->elements;
      int sum = 0;
      for(size_t index = 0; index < input->count; index++)
      {
        sum += input_elements[index];
      }
      result = INT2NUM(sum);
      break;
    }
    case nm_int:
    {
      int* input_elements = (int*)input->elements;
      int sum = 0;
      for(size_t index = 0; index < input->count; index++){
        sum += input_elements[index];
      }
      result = INT2NUM(sum);
      break;
    }
    case nm_float32:
    {
      float* input_elements = (float*)input->elements;
      float sum = 0;
      for(size_t index = 0; index < input->count; index++){
        sum += sin(input_elements[index]);
      }
      result = DBL2NUM((double)sum);
      break;
    }
    case nm_float64:
    {
      double* input_elements = (double*)input->elements;
      double sum = 0;
      for(size_t index = 0; index < input->count; index++){
        sum += input_elements[index];
      }
      result = DBL2NUM(sum);
      break;
    }
    case nm_complex32:
    {
      complex float* input_elements = (complex float*)input->elements;
      complex float sum = 0 + 0*I;
      for(size_t index = 0; index < input->count; index++){
        sum += input_elements[index];
      }
      double real = (double)creal(sum);
      double imag = (double)cimag(sum);
      result = rb_Complex(DBL2NUM(real), DBL2NUM(imag));
      break;
    }
    case nm_complex64:
    {
      complex double* input_elements = (complex double*)input->elements;
      complex double sum = 0 + 0*I;
      for(size_t index = 0; index < input->count; index++){
        sum += input_elements[index];
      }
      double real = creal(sum);
      double imag = cimag(sum);
      result = rb_Complex(DBL2NUM(real), DBL2NUM(imag));
      break;
    }
  }

  return result;
}

#define DEF_UNARY_RUBY_ACCESSOR(oper, name)                        \
static VALUE nm_##name(VALUE self) {                               \
  nmatrix* input;                                                  \
  Data_Get_Struct(self, nmatrix, input);                           \
                                                                   \
  nmatrix* result = ALLOC(nmatrix);                                \
  result->dtype = input->dtype;                                    \
  result->stype = input->stype;                                    \
  result->count = input->count;                                    \
  result->ndims = input->ndims;                                    \
  result->shape = ALLOC_N(size_t, result->ndims);                  \
                                                                   \
  for(size_t index = 0; index < result->ndims; index++){           \
    result->shape[index] = input->shape[index];                    \
  }                                                                \
  switch(result->dtype){                                           \
    case nm_bool:                                                   \
    {                                                              \
      bool* input_elements = (bool*)input->elements;                 \
      bool* result_elements = ALLOC_N(bool, result->count);          \
      for(size_t index = 0; index < input->count; index++){        \
        result_elements[index] = oper(input_elements[index]);      \
      }                                                            \
      result->elements = result_elements;                          \
      break;                                                       \
    }                                                              \
    case nm_int:                                                   \
    {                                                              \
      int* input_elements = (int*)input->elements;                 \
      int* result_elements = ALLOC_N(int, result->count);          \
      for(size_t index = 0; index < input->count; index++){        \
        result_elements[index] = oper(input_elements[index]);      \
      }                                                            \
      result->elements = result_elements;                          \
      break;                                                       \
    }                                                              \
    case nm_float32:                                               \
    {                                                              \
      float* input_elements = (float*)input->elements;             \
      float* result_elements = ALLOC_N(float, result->count);      \
      for(size_t index = 0; index < input->count; index++){        \
        result_elements[index] = oper(input_elements[index]);      \
      }                                                            \
      result->elements = result_elements;                          \
      break;                                                       \
    }                                                              \
    case nm_float64:                                               \
    {                                                              \
      double* input_elements = (double*)input->elements;           \
      double* result_elements = ALLOC_N(double, result->count);    \
      for(size_t index = 0; index < input->count; index++){        \
        result_elements[index] = oper(input_elements[index]);      \
      }                                                            \
      result->elements = result_elements;                          \
      break;                                                       \
    }                                                              \
    case nm_complex32:                                             \
    {                                                              \
      complex float* input_elements = (complex float*)input->elements;           \
      complex float* result_elements = ALLOC_N(complex float, result->count);    \
      for(size_t index = 0; index < input->count; index++){        \
        result_elements[index] = c##oper(input_elements[index]);      \
      }                                                            \
      result->elements = result_elements;                          \
      break;                                                       \
    }                                                              \
    case nm_complex64:                                               \
    {                                                              \
      complex double* input_elements = (complex double*)input->elements;           \
      complex double* result_elements = ALLOC_N(complex double, result->count);    \
      for(size_t index = 0; index < input->count; index++){        \
        result_elements[index] = c##oper(input_elements[index]);      \
      }                                                            \
      result->elements = result_elements;                          \
      break;                                                       \
    }                                                              \
  }                                                                \
  return Data_Wrap_Struct(NMatrix, NULL, nm_free, result);         \
}

DEF_UNARY_RUBY_ACCESSOR(cos, cos)
DEF_UNARY_RUBY_ACCESSOR(tan, tan)
DEF_UNARY_RUBY_ACCESSOR(asin, asin)
DEF_UNARY_RUBY_ACCESSOR(acos, acos)
DEF_UNARY_RUBY_ACCESSOR(atan, atan)
DEF_UNARY_RUBY_ACCESSOR(sinh, sinh)
DEF_UNARY_RUBY_ACCESSOR(cosh, cosh)
DEF_UNARY_RUBY_ACCESSOR(tanh, tanh)
DEF_UNARY_RUBY_ACCESSOR(asinh, asinh)
DEF_UNARY_RUBY_ACCESSOR(acosh, acosh)
DEF_UNARY_RUBY_ACCESSOR(atanh, atanh)
DEF_UNARY_RUBY_ACCESSOR(exp, exp)
DEF_UNARY_RUBY_ACCESSOR(log10, log10)
DEF_UNARY_RUBY_ACCESSOR(sqrt, sqrt)

#define DEF_UNARY_RUBY_ACCESSOR_NON_COMPLEX(oper, name)            \
static VALUE nm_##name(VALUE self) {                               \
  nmatrix* input;                                                  \
  Data_Get_Struct(self, nmatrix, input);                           \
                                                                   \
  nmatrix* result = ALLOC(nmatrix);                                \
  result->dtype = input->dtype;                                    \
  result->stype = input->stype;                                    \
  result->count = input->count;                                    \
  result->ndims = input->ndims;                                    \
  result->shape = ALLOC_N(size_t, result->ndims);                  \
                                                                   \
  for(size_t index = 0; index < result->ndims; index++){           \
    result->shape[index] = input->shape[index];                    \
  }                                                                \
  switch(result->dtype){                                           \
    case nm_bool:                                                   \
    {                                                              \
      bool* input_elements = (bool*)input->elements;                 \
      bool* result_elements = ALLOC_N(bool, result->count);          \
      for(size_t index = 0; index < input->count; index++){        \
        result_elements[index] = oper(input_elements[index]);      \
      }                                                            \
      result->elements = result_elements;                          \
      break;                                                       \
    }                                                              \
    case nm_int:                                                   \
    {                                                              \
      int* input_elements = (int*)input->elements;                 \
      int* result_elements = ALLOC_N(int, result->count);          \
      for(size_t index = 0; index < input->count; index++){        \
        result_elements[index] = oper(input_elements[index]);      \
      }                                                            \
      result->elements = result_elements;                          \
      break;                                                       \
    }                                                              \
    case nm_float32:                                               \
    {                                                              \
      float* input_elements = (float*)input->elements;             \
      float* result_elements = ALLOC_N(float, result->count);      \
      for(size_t index = 0; index < input->count; index++){        \
        result_elements[index] = oper(input_elements[index]);      \
      }                                                            \
      result->elements = result_elements;                          \
      break;                                                       \
    }                                                              \
    case nm_float64:                                               \
    {                                                              \
      double* input_elements = (double*)input->elements;           \
      double* result_elements = ALLOC_N(double, result->count);    \
      for(size_t index = 0; index < input->count; index++){        \
        result_elements[index] = oper(input_elements[index]);      \
      }                                                            \
      result->elements = result_elements;                          \
      break;                                                       \
    }                                                              \
    case nm_complex32:                                             \
    {                                                              \
      /* Not supported message */                                  \
    }                                                              \
    case nm_complex64:                                             \
    {                                                              \
      /* Not supported message */                                  \
    }                                                              \
  }                                                                \
  return Data_Wrap_Struct(NMatrix, NULL, nm_free, result);         \
}

DEF_UNARY_RUBY_ACCESSOR_NON_COMPLEX(log2, log2)
DEF_UNARY_RUBY_ACCESSOR_NON_COMPLEX(log1p, log1p)
DEF_UNARY_RUBY_ACCESSOR_NON_COMPLEX(erf, erf)
DEF_UNARY_RUBY_ACCESSOR_NON_COMPLEX(erfc, erfc)
DEF_UNARY_RUBY_ACCESSOR_NON_COMPLEX(cbrt, cbrt)
DEF_UNARY_RUBY_ACCESSOR_NON_COMPLEX(lgamma, lgamma)
DEF_UNARY_RUBY_ACCESSOR_NON_COMPLEX(tgamma, tgamma)
DEF_UNARY_RUBY_ACCESSOR_NON_COMPLEX(floor, floor)
DEF_UNARY_RUBY_ACCESSOR_NON_COMPLEX(ceil, ceil)