#include "ruby.h"
#include "stdio.h"
#include "cblas.h"
#include "math.h"

typedef struct NMATRIX_STRUCT
{
  size_t ndims;
  size_t count;
  size_t* shape;
  double* elements;
}nmatrix;

VALUE NMatrix = Qnil;

void Init_nmatrix();
VALUE nmatrix_init(int argc, VALUE* argv, VALUE self);
VALUE nm_get_dim(VALUE self);
VALUE nm_get_elements(VALUE self);
VALUE nm_get_shape(VALUE self);
VALUE nm_alloc(VALUE klass);
void nm_free(nmatrix* mat);

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
VALUE nm_inspect(VALUE self);

void Init_nmatrix() {
  NMatrix = rb_define_class("NMatrix", rb_cObject);

  rb_define_alloc_func(NMatrix, nm_alloc);
  rb_define_method(NMatrix, "initialize", nmatrix_init, -1);
  rb_define_method(NMatrix, "dim",      nm_get_dim, -1);
  rb_define_method(NMatrix, "shape",    nm_get_shape, 0);
  rb_define_method(NMatrix, "elements", nm_get_elements, 0);

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
  rb_define_method(NMatrix, "inspect", nm_inspect, 0);
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
    mat->elements = ALLOC_N(double, mat->count);
    for (size_t index = 0; index < mat->count; index++) {
      mat->elements[index] = (double)NUM2DBL(RARRAY_AREF(argv[1], index));
    }
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
  for (size_t index = 0; index < input->count; index++){
    array[index] = DBL2NUM(input->elements[index]);
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

VALUE nm_add(VALUE self, VALUE another){
  nmatrix* left;
  Data_Get_Struct(self, nmatrix, left);

  nmatrix* result = ALLOC(nmatrix);
  result->count = left->count;
  result->ndims = left->ndims;
  result->shape = ALLOC_N(size_t, result->ndims);

  for(size_t index = 0; index < result->ndims; index++){
    result->shape[index] = left->shape[index];
  }

  result->elements = ALLOC_N(double, result->count);

  if(RB_TYPE_P(another, T_FLOAT) || RB_TYPE_P(another, T_FIXNUM)){
    for(size_t index = 0; index < left->count; index++){
      result->elements[index] = left->elements[index] + NUM2DBL(another);
    }
  }
  else{
    nmatrix* right;
    Data_Get_Struct(another, nmatrix, right);

    for(size_t index = 0; index < left->count; index++){
      result->elements[index] = left->elements[index] + right->elements[index];
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
  result->count = left->count;                     \
  result->ndims = left->ndims;                     \
  result->shape = ALLOC_N(size_t, result->ndims);  \
                                                   \
  for(size_t index = 0; index < result->ndims; index++){             \
    result->shape[index] = left->shape[index];                       \
  }                                                                  \
                                                                     \
  if(RB_TYPE_P(another, T_FLOAT) || RB_TYPE_P(another, T_FIXNUM)){   \
    result->elements = ALLOC_N(double, result->count);               \
    for(size_t index = 0; index < left->count; index++){             \
      result->elements[index] = (left->elements[index]) oper + (NUM2DBL(another)); \
    }                                                                  \
  }                                                                    \
  else{                                                                \
    nmatrix* right;                                                    \
    Data_Get_Struct(another, nmatrix, right);                          \
    result->elements = ALLOC_N(double, result->count);                 \
    for(size_t index = 0; index < left->count; index++){               \
      result->elements[index] = (left->elements[index]) oper (right->elements[index]); \
    }                                                                                \
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
  result->count = input->count;
  result->ndims = input->ndims;
  result->shape = ALLOC_N(size_t, result->ndims);

  for(size_t index = 0; index < result->ndims; index++){
    result->shape[index] = input->shape[index];
  }

  result->elements = ALLOC_N(double, result->count);
  for(size_t index = 0; index < input->count; index++){
    result->elements[index] = sin(input->elements[index]);
  }

  return Data_Wrap_Struct(NMatrix, NULL, nm_free, result);
}

#define DEF_UNARY_RUBY_ACCESSOR(oper, name)                        \
static VALUE nm_##name(VALUE self) {                               \
  nmatrix* input;                                                  \
  Data_Get_Struct(self, nmatrix, input);                           \
                                                                   \
  nmatrix* result = ALLOC(nmatrix);                                \
  result->count = input->count;                                    \
  result->ndims = input->ndims;                                    \
  result->shape = ALLOC_N(size_t, result->ndims);                  \
                                                                   \
  for(size_t index = 0; index < result->ndims; index++){           \
    result->shape[index] = input->shape[index];                    \
  }                                                                \
                                                                   \
  result->elements = ALLOC_N(double, result->count);               \
  for(size_t index = 0; index < input->count; index++){            \
    result->elements[index] = oper(input->elements[index]);      \
  }                                                                \
                                                                   \
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
  nmatrix* input;
  Data_Get_Struct(self, nmatrix, input);

  return Qnil;
}

VALUE nm_accessor_set(int argc, VALUE* argv, VALUE self){
  nmatrix* input;
  Data_Get_Struct(self, nmatrix, input);

  return Qnil;
}

VALUE nm_get_rank(VALUE self, VALUE dim_val){
  nmatrix* input;
  Data_Get_Struct(self, nmatrix, input);
  size_t dim = NUM2LONG(dim_val);

  //get row

  nmatrix* result = ALLOC(nmatrix);
  result->count = input->shape[1];
  result->ndims = 2;
  result->shape = ALLOC_N(size_t, result->ndims);
  result->shape[0] = 1;
  result->shape[1] = input->shape[1];

  result->elements = ALLOC_N(double, input->shape[1]);

  for (size_t i = 0; i < input->shape[1]; ++i)
  {
    result->elements[i] = input->elements[input->shape[1]*dim + i];
  }

  return Data_Wrap_Struct(NMatrix, NULL, nm_free, result);
}

VALUE nm_get_dtype(VALUE self){
  return Qnil;
}

VALUE nm_inspect(VALUE self){
  char*  c = "Class: NMatrix";

  return rb_str_new_cstr(c);
}


#include "blas.c"
