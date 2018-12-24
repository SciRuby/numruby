#include "ruby.h"
#include "stdio.h"
#include "cblas.h"
#include "math.h"
#include "complex.h"

# define NM_NUM_DTYPES 2
# define NM_NUM_STYPES 2

typedef enum nm_dtype{
  nm_float32,
  nm_float64,
}nm_dtype;

const char* const DTYPE_NAMES[NM_NUM_DTYPES] = {
  "nm_float32",
  "nm_float64",
};

typedef enum nm_stype{
  nm_dense,
  nm_sparse
}nm_stype;

const char* const STYPE_NAMES[NM_NUM_STYPES] = {
  "nm_dense",
  "nm_sparse"
};

nm_dtype nm_dtype_from_rbsymbol(VALUE sym) {
  ID sym_id = SYM2ID(sym);

  for (size_t index = 0; index < NM_NUM_DTYPES; ++index) {
    if (sym_id == rb_intern(DTYPE_NAMES[index])) {
      return (nm_dtype)index;
    }
  }

  VALUE str = rb_any_to_s(sym);
  rb_raise(rb_eArgError, "invalid data type symbol (:%s) specified", RSTRING_PTR(str));
}

nm_dtype nm_stype_from_rbsymbol(VALUE sym) {
  ID sym_id = SYM2ID(sym);

  for (size_t index = 0; index < NM_NUM_STYPES; ++index) {
    if (sym_id == rb_intern(STYPE_NAMES[index])) {
      return (nm_stype)index;
    }
  }

  VALUE str = rb_any_to_s(sym);
  rb_raise(rb_eArgError, "invalid storage type symbol (:%s) specified", RSTRING_PTR(str));
}

typedef struct NMATRIX_STRUCT
{
  nm_dtype dtype;
  nm_stype stype;
  size_t ndims;
  size_t count;
  size_t* shape;
  void* elements;
}nmatrix;

VALUE NumRuby = Qnil;
VALUE NMatrix = Qnil;

void Init_nmatrix();

VALUE constant_nmatrix(int argc, VALUE* argv, double constant);
VALUE zeros_nmatrix(int argc, VALUE* argv);
VALUE ones_nmatrix(int argc, VALUE* argv);

VALUE nmatrix_init(int argc, VALUE* argv, VALUE self);
VALUE nm_get_dim(VALUE self);
VALUE nm_get_elements(VALUE self);
VALUE nm_get_shape(VALUE self);
VALUE nm_alloc(VALUE klass);
void nm_free(nmatrix* mat);

VALUE nm_eqeq(VALUE self, VALUE another);

VALUE nm_add(VALUE self, VALUE another);

#define DECL_ELEMENTWISE_RUBY_ACCESSOR(name)    VALUE nm_##name(VALUE self, VALUE another);

DECL_ELEMENTWISE_RUBY_ACCESSOR(subtract)
DECL_ELEMENTWISE_RUBY_ACCESSOR(multiply)
DECL_ELEMENTWISE_RUBY_ACCESSOR(divide)


VALUE nm_sin(VALUE self);

#define DECL_UNARY_RUBY_ACCESSOR(name)          static VALUE nm_##name(VALUE self);
DECL_UNARY_RUBY_ACCESSOR(cos)
DECL_UNARY_RUBY_ACCESSOR(tan)
DECL_UNARY_RUBY_ACCESSOR(asin)
DECL_UNARY_RUBY_ACCESSOR(acos)
DECL_UNARY_RUBY_ACCESSOR(atan)
DECL_UNARY_RUBY_ACCESSOR(sinh)
DECL_UNARY_RUBY_ACCESSOR(cosh)
DECL_UNARY_RUBY_ACCESSOR(tanh)
DECL_UNARY_RUBY_ACCESSOR(asinh)
DECL_UNARY_RUBY_ACCESSOR(acosh)
DECL_UNARY_RUBY_ACCESSOR(atanh)
DECL_UNARY_RUBY_ACCESSOR(exp)
DECL_UNARY_RUBY_ACCESSOR(log2)
DECL_UNARY_RUBY_ACCESSOR(log1p)
DECL_UNARY_RUBY_ACCESSOR(log10)
DECL_UNARY_RUBY_ACCESSOR(sqrt)
DECL_UNARY_RUBY_ACCESSOR(erf)
DECL_UNARY_RUBY_ACCESSOR(erfc)
DECL_UNARY_RUBY_ACCESSOR(cbrt)
DECL_UNARY_RUBY_ACCESSOR(lgamma)
DECL_UNARY_RUBY_ACCESSOR(tgamma)
DECL_UNARY_RUBY_ACCESSOR(floor)
DECL_UNARY_RUBY_ACCESSOR(ceil)

VALUE nm_dot(VALUE self, VALUE another);

VALUE nm_accessor_get(int argc, VALUE* argv, VALUE self);
VALUE nm_accessor_set(int argc, VALUE* argv, VALUE self);
VALUE nm_get_rank(VALUE self, VALUE dim);
VALUE nm_get_dtype(VALUE self);
VALUE nm_get_stype(VALUE self);
VALUE nm_inspect(VALUE self);

void Init_nmatrix() {
  NumRuby = rb_define_module("NumRuby");
  rb_define_singleton_method(NumRuby, "zeros",  zeros_nmatrix, -1);
  rb_define_singleton_method(NumRuby, "ones",   ones_nmatrix, -1);
  // rb_define_singleton_method(NumRuby, "matrix", nmatrix_init, -1);

  NMatrix = rb_define_class("NMatrix", rb_cObject);

  rb_define_alloc_func(NMatrix, nm_alloc);
  rb_define_method(NMatrix, "initialize", nmatrix_init, -1);
  rb_define_method(NMatrix, "dim",      nm_get_dim, -1);
  rb_define_method(NMatrix, "shape",    nm_get_shape, 0);
  rb_define_method(NMatrix, "elements", nm_get_elements, 0);
  rb_define_method(NMatrix, "dtype",    nm_get_dtype, 0);
  rb_define_method(NMatrix, "stype",    nm_get_stype, 0);

  rb_define_method(NMatrix, "==", nm_eqeq, 1);

  rb_define_method(NMatrix, "+", nm_add, 1);
  rb_define_method(NMatrix, "-", nm_subtract, 1);
  rb_define_method(NMatrix, "*", nm_multiply, 1);
  rb_define_method(NMatrix, "/", nm_divide, 1);

  rb_define_method(NMatrix, "sin", nm_sin, 0);
  rb_define_method(NMatrix, "cos", nm_cos, 0);
  rb_define_method(NMatrix, "tan", nm_tan, 0);
  rb_define_method(NMatrix, "asin", nm_asin, 0);
  rb_define_method(NMatrix, "acos", nm_acos, 0);
  rb_define_method(NMatrix, "atan", nm_atan, 0);
  rb_define_method(NMatrix, "sinh", nm_sinh, 0);
  rb_define_method(NMatrix, "cosh", nm_cosh, 0);
  rb_define_method(NMatrix, "tanh", nm_tanh, 0);
  rb_define_method(NMatrix, "asinh", nm_asinh, 0);
  rb_define_method(NMatrix, "acosh", nm_acosh, 0);
  rb_define_method(NMatrix, "atanh", nm_atanh, 0);
  rb_define_method(NMatrix, "exp", nm_exp, 0);
  rb_define_method(NMatrix, "log2", nm_log2, 0);
  rb_define_method(NMatrix, "log1p", nm_log1p, 0);
  rb_define_method(NMatrix, "log10", nm_log10, 0);
  rb_define_method(NMatrix, "sqrt", nm_sqrt, 0);
  rb_define_method(NMatrix, "erf", nm_erf, 0);
  rb_define_method(NMatrix, "erfc", nm_erfc, 0);
  rb_define_method(NMatrix, "cbrt", nm_cbrt, 0);
  rb_define_method(NMatrix, "lgamma", nm_lgamma, 0);
  rb_define_method(NMatrix, "tgamma", nm_tgamma, 0);
  rb_define_method(NMatrix, "floor", nm_floor, 0);
  rb_define_method(NMatrix, "ceil", nm_ceil, 0);

  rb_define_method(NMatrix, "dot", nm_dot, 1);

  rb_define_method(NMatrix, "[]", nm_accessor_get, -1);
  rb_define_method(NMatrix, "[]=", nm_accessor_set, -1);
  rb_define_method(NMatrix, "row", nm_get_rank, 1);
  rb_define_method(NMatrix, "dtype", nm_get_dtype, 0);
  // rb_define_method(NMatrix, "inspect", nm_inspect, 0);
}

VALUE zeros_nmatrix(int argc, VALUE* argv){
  return constant_nmatrix(argc, argv, 0);
}

VALUE ones_nmatrix(int argc, VALUE* argv){
  return constant_nmatrix(argc, argv, 1);
}

VALUE constant_nmatrix(int argc, VALUE* argv, double constant){
  nmatrix* mat = ALLOC(nmatrix);
  mat->stype = nm_dense;
  mat->dtype = nm_float64;
  mat->ndims = 2;
  mat->count = 1;
  mat->shape = ALLOC_N(size_t, mat->ndims);
  for (size_t index = 0; index < mat->ndims; index++) {
    mat->shape[index] = (size_t)FIX2LONG(RARRAY_AREF(argv[0], index));
    mat->count *= mat->shape[index];
  }

  double *elements = ALLOC_N(double, mat->count);
  for (size_t index = 0; index < mat->count; index++) {
    elements[index] = constant;
  }

  mat->elements = elements;

  return Data_Wrap_Struct(NMatrix, NULL, nm_free, mat);
}

VALUE nmatrix_init(int argc, VALUE* argv, VALUE self){
  nmatrix* mat;
  Data_Get_Struct(self, nmatrix, mat);

  if(argc > 0){
    mat->ndims = 2;
    mat->count = 1;
    mat->shape = ALLOC_N(size_t, mat->ndims);
    for (size_t index = 0; index < mat->ndims; index++) {
      mat->shape[index] = (size_t)FIX2LONG(RARRAY_AREF(argv[0], index));
      mat->count *= mat->shape[index];
    }
    mat->dtype = (argc > 2) ? nm_dtype_from_rbsymbol(argv[2]) : nm_float64;
    switch (mat->dtype) {
      case nm_float64:
      {
        double* elements = ALLOC_N(double, mat->count);
        for (size_t index = 0; index < mat->count; index++) {
          elements[index] = (double)NUM2DBL(RARRAY_AREF(argv[1], index));
        }
        mat->elements = elements;
        break;
      }
      case nm_float32:
      {
        float* elements = ALLOC_N(float, mat->count);
        for (size_t index = 0; index < mat->count; index++) {
          elements[index] = (float)NUM2DBL(RARRAY_AREF(argv[1], index));
        }
        mat->elements = elements;
        break;
      }
    }

    mat->stype = (argc > 3) ? nm_stype_from_rbsymbol(argv[3]) : nm_dense;
  }

  return self;
}

VALUE nm_alloc(VALUE klass)
{
  nmatrix* mat = ALLOC(nmatrix);

  return Data_Wrap_Struct(klass, NULL, nm_free, mat);
}

void nm_free(nmatrix* mat){
  xfree(mat);
}

VALUE nm_get_dim(VALUE self){
  return INT2NUM(2);
}

VALUE nm_get_elements(VALUE self){
  nmatrix* input;

  Data_Get_Struct(self, nmatrix, input);

  VALUE* array = ALLOC_N(VALUE, input->count);

  switch (input->dtype) {
    case nm_float64:
    {
      double* elements = (double*)input->elements;
      for (size_t index = 0; index < input->count; index++){
        array[index] = DBL2NUM(elements[index]);
      }
      break;
    }
    case nm_float32:
    {
      float* elements = (float*)input->elements;
      for (size_t index = 0; index < input->count; index++){
        array[index] = DBL2NUM(elements[index]);
      }
      break;
    }
  }

  return rb_ary_new4(input->count, array);
}

VALUE nm_get_shape(VALUE self){
  nmatrix* input;

  Data_Get_Struct(self, nmatrix, input);

  VALUE* array = ALLOC_N(VALUE, input->ndims);
  for (size_t index = 0; index < input->ndims; index++){
    array[index] = LONG2NUM(input->shape[index]);
  }

  return rb_ary_new4(input->ndims, array);
}

VALUE nm_get_dtype(VALUE self){
  nmatrix* nmat;
  Data_Get_Struct(self, nmatrix, nmat);

  return rb_str_new_cstr(DTYPE_NAMES[nmat->dtype]);
}

VALUE nm_get_stype(VALUE self){
  nmatrix* nmat;
  Data_Get_Struct(self, nmatrix, nmat);

  return rb_str_new_cstr(STYPE_NAMES[nmat->stype]);
}

VALUE nm_eqeq(VALUE self, VALUE another){
  nmatrix* left;
  nmatrix* right;
  Data_Get_Struct(self, nmatrix, left);
  Data_Get_Struct(another, nmatrix, right);

  double* left_elements = (double*)left->elements;
  double* right_elements = (double*)right->elements;

  for(size_t index = 0; index < left->count; index++){
    if(fabs(left_elements[index] - right_elements[index]) > 1e-3){
      return Qfalse;
    }
  }

  return Qtrue;
}

VALUE nm_add(VALUE self, VALUE another){
  nmatrix* left;
  Data_Get_Struct(self, nmatrix, left);

  nmatrix* result = ALLOC(nmatrix);
  result->dtype = left->dtype;
  result->stype = left->stype;
  result->count = left->count;
  result->ndims = left->ndims;
  result->shape = ALLOC_N(size_t, result->ndims);

  for(size_t index = 0; index < result->ndims; index++){
    result->shape[index] = left->shape[index];
  }

  switch (result->dtype) {
    case nm_float64:
    {
      double* left_elements = (double*)left->elements;
      double* result_elements = ALLOC_N(double, result->count);
      if(RB_TYPE_P(another, T_FLOAT) || RB_TYPE_P(another, T_FIXNUM)){
        for(size_t index = 0; index < left->count; index++){
          result_elements[index] = left_elements[index] + NUM2DBL(another);
        }
      }
      else{
        nmatrix* right;
        Data_Get_Struct(another, nmatrix, right);
        double* right_elements = (double*)right->elements;

        for(size_t index = 0; index < left->count; index++){
          result_elements[index] = left_elements[index] + right_elements[index];
        }
      }
      result->elements = result_elements;
      break;
    }
    case nm_float32:
    {
      float* left_elements = (float*)left->elements;
      float* result_elements = ALLOC_N(float, result->count);
      if(RB_TYPE_P(another, T_FLOAT) || RB_TYPE_P(another, T_FIXNUM)){
        for(size_t index = 0; index < left->count; index++){
          result_elements[index] = left_elements[index] + NUM2DBL(another);
        }
      }
      else{
        nmatrix* right;
        Data_Get_Struct(another, nmatrix, right);
        float* right_elements = (float*)right->elements;

        for(size_t index = 0; index < left->count; index++){
          result_elements[index] = left_elements[index] + right_elements[index];
        }
      }
      result->elements = result_elements;
      break;
    }
  }
  return Data_Wrap_Struct(NMatrix, NULL, nm_free, result);
}

#define DEF_ELEMENTWISE_RUBY_ACCESSOR(name, oper)  \
VALUE nm_##name(VALUE self, VALUE another){        \
  nmatrix* left;                                   \
  Data_Get_Struct(self, nmatrix, left);            \
                                                   \
  nmatrix* result = ALLOC(nmatrix);                \
  result->dtype = left->dtype;                     \
  result->stype = left->stype;                     \
  result->count = left->count;                     \
  result->ndims = left->ndims;                     \
  result->shape = ALLOC_N(size_t, result->ndims);  \
                                                   \
  for(size_t index = 0; index < result->ndims; index++){             \
    result->shape[index] = left->shape[index];                       \
  }                                                                  \
                                                                     \
  switch (result->dtype) {                                                       \
    case nm_float64:                                                             \
    {                                                                            \
      double* left_elements = (double*)left->elements;                           \
      double* result_elements = ALLOC_N(double, result->count);                  \
      if(RB_TYPE_P(another, T_FLOAT) || RB_TYPE_P(another, T_FIXNUM)){           \
        for(size_t index = 0; index < left->count; index++){                     \
          result_elements[index] = left_elements[index] + NUM2DBL(another);      \
        }                                                                        \
      }                                                                          \
      else{                                                                      \
        nmatrix* right;                                                          \
        Data_Get_Struct(another, nmatrix, right);                                \
        double* right_elements = (double*)right->elements;                       \
                                                                                 \
        for(size_t index = 0; index < left->count; index++){                     \
          result_elements[index] = (left_elements[index]) oper (right_elements[index]); \
        }                                                                        \
      }                                                                          \
      result->elements = result_elements;                                        \
      break;                                                                     \
    }                                                                            \
    case nm_float32:                                                             \
    {                                                                            \
      float* left_elements = (float*)left->elements;                             \
      float* result_elements = ALLOC_N(float, result->count);                    \
      if(RB_TYPE_P(another, T_FLOAT) || RB_TYPE_P(another, T_FIXNUM)){           \
        for(size_t index = 0; index < left->count; index++){                     \
          result_elements[index] = left_elements[index] + NUM2DBL(another);      \
        }                                                                        \
      }                                                                          \
      else{                                                                      \
        nmatrix* right;                                                          \
        Data_Get_Struct(another, nmatrix, right);                                \
        float* right_elements = (float*)right->elements;                         \
                                                                                 \
        for(size_t index = 0; index < left->count; index++){                       \
          result_elements[index] = left_elements[index] + right_elements[index];   \
        }                                                                          \
      }                                                                            \
      result->elements = result_elements;                                          \
      break;                                                                       \
    }                                                                              \
  }                                                                                \
  return Data_Wrap_Struct(NMatrix, NULL, nm_free, result);                         \
}

DEF_ELEMENTWISE_RUBY_ACCESSOR(subtract, -)
DEF_ELEMENTWISE_RUBY_ACCESSOR(multiply, *)
DEF_ELEMENTWISE_RUBY_ACCESSOR(divide, /)

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
    case nm_float32:
    {
      float* input_elements = (float*)input->elements;
      float* result_elements = ALLOC_N(float, result->count);
      for(size_t index = 0; index < input->count; index++){
        result_elements[index] = sin(input_elements[index]);
      }
      result->elements = result_elements;
    }
    case nm_float64:
    {
      double* input_elements = (double*)input->elements;
      double* result_elements = ALLOC_N(double, result->count);
      for(size_t index = 0; index < input->count; index++){
        result_elements[index] = sin(input_elements[index]);
      }
      result->elements = result_elements;
    }
  }

  return Data_Wrap_Struct(NMatrix, NULL, nm_free, result);
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
DEF_UNARY_RUBY_ACCESSOR(log2, log2)
DEF_UNARY_RUBY_ACCESSOR(log1p, log1p)
DEF_UNARY_RUBY_ACCESSOR(log10, log10)
DEF_UNARY_RUBY_ACCESSOR(sqrt, sqrt)
DEF_UNARY_RUBY_ACCESSOR(erf, erf)
DEF_UNARY_RUBY_ACCESSOR(erfc, erfc)
DEF_UNARY_RUBY_ACCESSOR(cbrt, cbrt)
DEF_UNARY_RUBY_ACCESSOR(lgamma, lgamma)
DEF_UNARY_RUBY_ACCESSOR(tgamma, tgamma)
DEF_UNARY_RUBY_ACCESSOR(floor, floor)
DEF_UNARY_RUBY_ACCESSOR(ceil, ceil)

VALUE nm_accessor_get(int argc, VALUE* argv, VALUE self){
  nmatrix* nmat;
  Data_Get_Struct(self, nmatrix, nmat);

  size_t index = (size_t)FIX2LONG(argv[0]) * nmat->shape[1] +  (size_t)FIX2LONG(argv[1]);
  double* elements = (double*)nmat->elements;
  double val = elements[index];

  return DBL2NUM(val);
}

VALUE nm_accessor_set(int argc, VALUE* argv, VALUE self){
  nmatrix* nmat;
  Data_Get_Struct(self, nmatrix, nmat);

  size_t index = (size_t)FIX2LONG(argv[0]) * nmat->shape[1] +  (size_t)FIX2LONG(argv[1]);

  double* elements = (double*)nmat->elements;
  elements[index] = NUM2DBL(argv[2]);

  nmat->elements = elements;
  return argv[2];
}

VALUE nm_get_rank(VALUE self, VALUE dim_val){
  nmatrix* input;
  Data_Get_Struct(self, nmatrix, input);
  double* input_elements = (double*)input->elements;

  size_t dim = NUM2LONG(dim_val);

  //get row

  nmatrix* result = ALLOC(nmatrix);
  result->dtype = input->dtype;
  result->stype = input->stype;
  result->count = input->shape[1];
  result->ndims = 2;
  result->shape = ALLOC_N(size_t, result->ndims);
  result->shape[0] = 1;
  result->shape[1] = input->shape[1];

  double* result_elements = ALLOC_N(double, input->shape[1]);

  for (size_t i = 0; i < input->shape[1]; ++i)
  {
    result_elements[i] = input_elements[input->shape[1]*dim + i];
  }
  result->elements = result_elements;

  return Data_Wrap_Struct(NMatrix, NULL, nm_free, result);
}

VALUE nm_inspect(VALUE self){
  const char* c = "Class: NMatrix";

  return rb_str_new_cstr(c);
}


#include "blas.c"
