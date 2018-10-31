#include "ruby.h"
#include "stdio.h"

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
VALUE nm_get_elements(VALUE self);
VALUE nm_get_shape(VALUE self);
VALUE nm_alloc(VALUE klass);
void nm_free(nmatrix* mat);

VALUE nm_add(VALUE self, VALUE another);

#define DECL_ELEMENTWISE_RUBY_ACCESSOR(name)    VALUE nm_##name(VALUE self, VALUE another);

DECL_ELEMENTWISE_RUBY_ACCESSOR(subtract)
DECL_ELEMENTWISE_RUBY_ACCESSOR(multiply)
DECL_ELEMENTWISE_RUBY_ACCESSOR(divide)


void Init_nmatrix() {
  NMatrix = rb_define_class("NMatrix", rb_cObject);

  rb_define_alloc_func(NMatrix, nm_alloc);
  rb_define_method(NMatrix, "initialize", nmatrix_init, -1);
  rb_define_method(NMatrix, "shape", nm_get_shape, 0);
  rb_define_method(NMatrix, "elements", nm_get_elements, 0);

  rb_define_method(NMatrix, "+", nm_add, 1);
  rb_define_method(NMatrix, "-", nm_subtract, 1);
  rb_define_method(NMatrix, "*", nm_multiply,1);
  rb_define_method(NMatrix, "/", nm_divide,1);
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
  nmatrix* right;
  Data_Get_Struct(self, nmatrix, left);
  Data_Get_Struct(another, nmatrix, right);

  nmatrix* result = ALLOC(nmatrix);
  result->count = left->count;
  result->ndims = left->ndims;
  result->shape = ALLOC_N(size_t, result->ndims);

  for(size_t index = 0; index < result->ndims; index++){
    result->shape[index] = left->shape[index];
  }

  result->elements = ALLOC_N(double, result->count);
  for(size_t index = 0; index < left->count; index++){
    result->elements[index] = left->elements[index] + right->elements[index];
  }

  return Data_Wrap_Struct(NMatrix, NULL, nm_free, result);
}

#define DEF_ELEMENTWISE_RUBY_ACCESSOR(name, oper)  \
VALUE nm_##name(VALUE self, VALUE another){        \
  nmatrix* left;                                   \
  nmatrix* right;                                  \
  Data_Get_Struct(self, nmatrix, left);            \
  Data_Get_Struct(another, nmatrix, right);        \
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
  result->elements = ALLOC_N(double, result->count);                 \
  for(size_t index = 0; index < left->count; index++){               \
    result->elements[index] = (left->elements[index]) oper (right->elements[index]); \
  }                                                                                \
                                                                                  \
  return Data_Wrap_Struct(NMatrix, NULL, nm_free, result);                         \
}

DEF_ELEMENTWISE_RUBY_ACCESSOR(subtract, -)
DEF_ELEMENTWISE_RUBY_ACCESSOR(multiply, *)
DEF_ELEMENTWISE_RUBY_ACCESSOR(divide, /)
