#include "ruby.h"
#include "stdio.h"
#include "cblas.h"
#include "lapacke.h"
#include "math.h"
#include "complex.h"
#include "stdbool.h"

# define NM_NUM_DTYPES 6
# define NM_NUM_STYPES 2
# define NM_NUM_SPARSE_TYPES 4

#define max(a,b) \
 ({ __typeof__ (a) _a = (a); \
     __typeof__ (b) _b = (b); \
   _a > _b ? _a : _b; })

#define min(a,b) \
 ({ __typeof__ (a) _a = (a); \
     __typeof__ (b) _b = (b); \
   _a < _b ? _a : _b; })

// data types
typedef enum nm_dtype{
  nm_bool,
  nm_int,
  nm_float32,
  nm_float64,
  nm_complex32,
  nm_complex64
}nm_dtype;

const char* const DTYPE_NAMES[NM_NUM_DTYPES] = {
  "nm_bool",
  "nm_int",
  "nm_float32",
  "nm_float64",
  "nm_complex32",
  "nm_complex64"
};

// storage types
typedef enum nm_stype{
  nm_dense,
  nm_sparse
}nm_stype;

const char* const STYPE_NAMES[NM_NUM_STYPES] = {
  "nm_dense",
  "nm_sparse"
};

typedef struct COO_NMATRIX{
  size_t count;
  void* elements;
  size_t* ia;
  size_t* ja;
}coo_nmatrix;

typedef struct CSC_NMATRIX{
  size_t count;
  void* elements;
  size_t* ia;
  size_t* ja;
}csc_nmatrix;

typedef struct CSR_NMATRIX{
  size_t count;
  void* elements;
  size_t* ia;
  size_t* ja;
}csr_nmatrix;

typedef struct DIAG_MATRIX{
  size_t count;
  void* elements;
  size_t* offset;
}diag_nmatrix;

typedef struct SPARSE_STORAGE{
  csr_nmatrix* csr;
}sparse_storage;

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
  sparse_storage* sp;
}nmatrix;

nmatrix* nmatrix_new(
  nm_dtype dtype,
  nm_stype stype,
  size_t ndims,
  size_t count,
  size_t* shape,
  void* elements
  ) {
  nmatrix* matrix = ALLOC(nmatrix);
  matrix->dtype = dtype;
  matrix->stype = stype;
  matrix->ndims = ndims;
  matrix->count = count;

  matrix->shape = ALLOC_N(size_t, matrix->ndims);
  if(shape != NULL) {
    for(size_t i = 0; i < ndims; ++i) {
      matrix->shape[i] = shape[i];
    }
  }

  if(elements == NULL) {
    return matrix;
  }

  switch(dtype) {
    case nm_bool:
    {
      bool* temp_elements = (bool*)elements;
      bool* matrix_elements = ALLOC_N(bool, matrix->count);
      for(size_t i = 0; i < count; ++i) {
        matrix_elements[i] = temp_elements[i];
      }
      matrix->elements = matrix_elements;
      break;
    }
    case nm_int:
    {
      int* temp_elements = (int*)elements;
      int* matrix_elements = ALLOC_N(int, matrix->count);
      for(size_t i = 0; i < count; ++i) {
        matrix_elements[i] = temp_elements[i];
      }
      matrix->elements = matrix_elements;
      break;
    }
    case nm_float32:
    {
      float* temp_elements = (float*)elements;
      float* matrix_elements = ALLOC_N(float, matrix->count);
      for(size_t i = 0; i < count; ++i) {
        matrix_elements[i] = temp_elements[i];
      }
      matrix->elements = matrix_elements;
      break;
    }
    case nm_float64:
    {
      double* temp_elements = (double*)elements;
      double* matrix_elements = ALLOC_N(double, matrix->count);
      for(size_t i = 0; i < count; ++i) {
        matrix_elements[i] = temp_elements[i];
      }
      matrix->elements = matrix_elements;
      break;
    }
    case nm_complex32:
    {
      float complex* temp_elements = (float complex*)elements;
      float complex* matrix_elements = ALLOC_N(float complex, matrix->count);
      for(size_t i = 0; i < count; ++i) {
        matrix_elements[i] = temp_elements[i];
      }
      matrix->elements = matrix_elements;
      break;
    }
    case nm_complex64:
    {
      double complex* temp_elements = (double complex*)elements;
      double complex* matrix_elements = ALLOC_N(double complex, matrix->count);
      for(size_t i = 0; i < count; ++i) {
        matrix_elements[i] = temp_elements[i];
      }
      matrix->elements = matrix_elements;
      break;
    }
  }

  return matrix;
}

nmatrix* matrix_copy(nmatrix* original_matrix) {
  nmatrix* matrix = ALLOC(nmatrix);
  matrix->dtype = original_matrix->dtype;
  matrix->stype = original_matrix->stype;
  matrix->ndims = original_matrix->ndims;
  matrix->count = original_matrix->count;

  matrix->shape = ALLOC_N(size_t, matrix->ndims);
  if(original_matrix->shape != NULL) {
    for(size_t i = 0; i < original_matrix->ndims; ++i) {
      matrix->shape[i] = original_matrix->shape[i];
    }
  }

  if(original_matrix->elements == NULL) {
    return matrix;
  }

  switch(original_matrix->dtype) {
    case nm_bool:
    {
      bool* temp_elements = (bool*)original_matrix->elements;
      bool* matrix_elements = ALLOC_N(bool, matrix->count);
      for(size_t i = 0; i < original_matrix->count; ++i) {
        matrix_elements[i] = temp_elements[i];
      }
      matrix->elements = matrix_elements;
      break;
    }
    case nm_int:
    {
      int* temp_elements = (int*)original_matrix->elements;
      int* matrix_elements = ALLOC_N(int, matrix->count);
      for(size_t i = 0; i < original_matrix->count; ++i) {
        matrix_elements[i] = temp_elements[i];
      }
      matrix->elements = matrix_elements;
      break;
    }
    case nm_float32:
    {
      float* temp_elements = (float*)original_matrix->elements;
      float* matrix_elements = ALLOC_N(float, matrix->count);
      for(size_t i = 0; i < original_matrix->count; ++i) {
        matrix_elements[i] = temp_elements[i];
      }
      matrix->elements = matrix_elements;
      break;
    }
    case nm_float64:
    {
      double* temp_elements = (double*)original_matrix->elements;
      double* matrix_elements = ALLOC_N(double, matrix->count);
      for(size_t i = 0; i < original_matrix->count; ++i) {
        matrix_elements[i] = temp_elements[i];
      }
      matrix->elements = matrix_elements;
      break;
    }
    case nm_complex32:
    {
      float complex* temp_elements = (float complex*)original_matrix->elements;
      float complex* matrix_elements = ALLOC_N(float complex, matrix->count);
      for(size_t i = 0; i < original_matrix->count; ++i) {
        matrix_elements[i] = temp_elements[i];
      }
      matrix->elements = matrix_elements;
      break;
    }
    case nm_complex64:
    {
      double complex* temp_elements = (double complex*)original_matrix->elements;
      double complex* matrix_elements = ALLOC_N(double complex, matrix->count);
      for(size_t i = 0; i < original_matrix->count; ++i) {
        matrix_elements[i] = temp_elements[i];
      }
      matrix->elements = matrix_elements;
      break;
    }
  }

  return matrix;
}

typedef enum nm_sparse_type{
  coo,
  csc,
  csr,
  dia
}nm_sparse_type;

const char* const SPARSE_TYPE_NAMES[NM_NUM_SPARSE_TYPES] = {
  "coo",
  "csc",
  "csr",
  "dia"
};

nm_sparse_type nm_sparse_type_from_rbsymbol(VALUE sym) {
  ID sym_id = SYM2ID(sym);

  for (size_t index = 0; index < NM_NUM_SPARSE_TYPES; ++index) {
    if (sym_id == rb_intern(SPARSE_TYPE_NAMES[index])) {
      return (nm_sparse_type)index;
    }
  }

  VALUE str = rb_any_to_s(sym);
  rb_raise(rb_eArgError, "invalid storage type symbol (:%s) specified", RSTRING_PTR(str));
}

typedef struct SPARSE_NMATRIX_STRUCT{
  nm_dtype dtype;
  nm_sparse_type sptype;
  size_t ndims;
  size_t count; //data count
  size_t* shape;
  coo_nmatrix* coo;
  csr_nmatrix* csr;
  csc_nmatrix* csc;
  diag_nmatrix* diag;
}sparse_nmatrix;

VALUE NumRuby = Qnil;
VALUE Lapack = Qnil;
VALUE Blas = Qnil;
VALUE DataTypeError = Qnil;
VALUE ShapeError = Qnil;
VALUE NMatrix = Qnil;
VALUE SparseNMatrix = Qnil;

void Init_nmatrix();

VALUE average_nmatrix(int argc, VALUE* argv);
VALUE constant_nmatrix(int argc, VALUE* argv, double constant);
VALUE zeros_nmatrix(int argc, VALUE* argv);
VALUE ones_nmatrix(int argc, VALUE* argv);
VALUE nm_broadcast_to(int argc, VALUE* argv);
//VALUE nm_broadcast_arrays(int argc, VALUE* argv)

VALUE nmatrix_init(int argc, VALUE* argv, VALUE self);
VALUE nm_get_dim(VALUE self);
VALUE nm_get_elements(VALUE self);
VALUE nm_get_shape(VALUE self);
VALUE nm_alloc(VALUE klass);
void nm_free(nmatrix* mat);

VALUE nm_each(VALUE self);
VALUE nm_each_with_indices(VALUE self);
//VALUE nm_each_stored_with_indices(VALUE self);
//VALUE nm_each_ordered_stored_with_indices(VALUE self);
//VALUE nm_map_stored(VALUE self);
VALUE nm_each_rank(VALUE self, VALUE dimension_idx);
VALUE nm_each_row(VALUE self);
VALUE nm_each_column(VALUE self);
VALUE nm_each_layer(VALUE self);

//VALUE nm_get_row(VALUE self, VALUE row_number);
//VALUE nm_get_column(VALUE self, VALUE column_number);

VALUE nm_eqeq(VALUE self, VALUE another);
VALUE nm_gt(  VALUE self, VALUE another);
VALUE nm_gteq(VALUE self, VALUE another);
VALUE nm_lt(  VALUE self, VALUE another);
VALUE nm_lteq(VALUE self, VALUE another);
VALUE nm_add( VALUE self, VALUE another);

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
VALUE nm_norm2(VALUE self);
void sgetrf(const float* arr, const size_t cols, const size_t rows, int* ipiv, float* arr2);
void dgetrf(const double* arr, const size_t cols, const size_t rows, int* ipiv, double* arr2);
void cgetrf(const float complex* arr, const size_t cols, const size_t rows, int* ipiv, float  complex* arr2);
void zgetrf(const double complex* arr, const size_t cols, const size_t rows, int* ipiv, double  complex* arr2);
VALUE nm_invert(VALUE self);
VALUE nm_solve(VALUE self, VALUE rhs_val);
VALUE nm_det(VALUE self);
VALUE nm_least_square(VALUE self, VALUE rhs_val);
VALUE nm_pinv(VALUE self);
VALUE nm_kronecker_prod(VALUE self);
VALUE nm_eig(VALUE self);
VALUE nm_eigh(VALUE self);
VALUE nm_eigvalsh(VALUE self);
VALUE nm_lu(VALUE self);
VALUE nm_lu_factor(VALUE self);
VALUE nm_lu_solve(VALUE self, VALUE rhs_val);
VALUE nm_svd(VALUE self);
VALUE nm_svdvals(VALUE self);
VALUE nm_diagsvd(VALUE self);

// LAPACK routines

VALUE nm_geqrf(int argc, VALUE* argv);
VALUE nm_orgqr(int argc, VALUE* argv);
VALUE nm_geqp3(int argc, VALUE* argv);
VALUE nm_potrf(int argc, VALUE* argv);
VALUE nm_potrs(int argc, VALUE* argv);
VALUE nm_gesdd(int argc, VALUE* argv);
VALUE nm_getrf(int argc, VALUE* argv);
VALUE nm_getrs(int argc, VALUE* argv);
VALUE nm_ggev(int argc, VALUE* argv);
VALUE nm_geev(int argc, VALUE* argv);
VALUE nm_heevr(int argc, VALUE* argv);
VALUE nm_syevr(int argc, VALUE* argv);
VALUE nm_hegvx(int argc, VALUE* argv);
VALUE nm_sygvx(int argc, VALUE* argv);
VALUE nm_hegvd(int argc, VALUE* argv);
VALUE nm_sygvd(int argc, VALUE* argv);
VALUE nm_hegv(int argc, VALUE* argv);
VALUE nm_sygv(int argc, VALUE* argv);
VALUE nm_getri(int argc, VALUE* argv);
VALUE nm_gelss(int argc, VALUE* argv);
VALUE nm_posv(int argc, VALUE* argv);
VALUE nm_gesv(int argc, VALUE* argv);
VALUE nm_lange(int argc, VALUE* argv);
  
VALUE nm_orth(VALUE self);
VALUE nm_cholesky(VALUE self);
VALUE nm_cholesky_solve(VALUE self);

VALUE nm_accessor_get(int argc, VALUE* argv, VALUE self);
VALUE nm_accessor_set(int argc, VALUE* argv, VALUE self);
VALUE nm_get_rank(VALUE self, VALUE dim);
VALUE nm_get_dtype(VALUE self);
VALUE nm_get_stype(VALUE self);
VALUE nm_inspect(VALUE self);

// Sparse Matrix

VALUE nm_sparse_alloc(VALUE klass);
void  nm_sparse_free(csr_nmatrix* mat);

VALUE coo_sparse_nmatrix_init(int argc, VALUE* argv);
VALUE csr_sparse_nmatrix_init(int argc, VALUE* argv);
VALUE csc_sparse_nmatrix_init(int argc, VALUE* argv);
VALUE dia_sparse_nmatrix_init(int argc, VALUE* argv);
VALUE nm_sparse_get_dtype(VALUE self);
VALUE nm_sparse_get_shape(VALUE self);
VALUE nm_sparse_to_array(VALUE self);

/*
 * Sparse matrix to NMatrix
 *
 * @return NMatrix
 */
VALUE nm_sparse_to_nmatrix(VALUE self);

void get_dense_from_coo(const void* data_t, const size_t rows,
                       const size_t cols, const size_t* ia,
                       const size_t* ja, void* elements_t, nm_dtype);
void get_dense_from_csc(const void* data_t, const size_t rows,
                       const size_t cols, const size_t* ia,
                       const size_t* ja, void* elements_t, nm_dtype);
void get_dense_from_csr(const void* data_t, const size_t rows,
                       const size_t cols, const size_t* ia,
                       const size_t* ja, void* elements_t, nm_dtype);
void get_dense_from_dia(const void* data_t, const size_t rows,
                       const size_t cols, const size_t* offset,
                       void* elements_t, nm_dtype);

//forwards for internally used functions
void get_slice(nmatrix* nmat, size_t* lower, size_t* upper, nmatrix* slice);
size_t get_index(nmatrix* nmat, VALUE* indices);


void Init_nmatrix() {

  ///////////////////////
  // Class Definitions //
  ///////////////////////

  NumRuby = rb_define_module("NumRuby");
  rb_define_singleton_method(NumRuby, "average",  average_nmatrix, -1);
  rb_define_singleton_method(NumRuby, "zeros",  zeros_nmatrix, -1);
  rb_define_singleton_method(NumRuby, "ones",   ones_nmatrix, -1);
  // rb_define_singleton_method(NumRuby, "matrix", nmatrix_init, -1);
  rb_define_singleton_method(NumRuby, "broadcast_to", nm_broadcast_to, -1);
  //rb_define_singleton_method(NumRuby, "broadcast_arrays", nm_broadcast_arrays, -1);

  Lapack = rb_define_module_under(NumRuby, "Lapack");
  rb_define_singleton_method(Lapack, "geqrf", nm_geqrf, -1);
  rb_define_singleton_method(Lapack, "orgqr", nm_orgqr, -1);
  rb_define_singleton_method(Lapack, "geqp3", nm_geqp3, -1);
  rb_define_singleton_method(Lapack, "potrf", nm_potrf, -1);
  rb_define_singleton_method(Lapack, "potrs", nm_potrs, -1);
  rb_define_singleton_method(Lapack, "gesdd", nm_gesdd, -1);
  rb_define_singleton_method(Lapack, "getrf", nm_getrf, -1);
  rb_define_singleton_method(Lapack, "getrs", nm_getrs, -1);
  rb_define_singleton_method(Lapack, "getri", nm_getri, -1);
  rb_define_singleton_method(Lapack, "gelss", nm_gelss, -1);
  rb_define_singleton_method(Lapack, "posv", nm_posv, -1);
  rb_define_singleton_method(Lapack, "gesv", nm_gesv, -1);
  rb_define_singleton_method(Lapack, "lange", nm_lange, -1);

  Blas = rb_define_module("Blas");

  /*
   * Exception raised when there's a problem with data.
   */
  DataTypeError = rb_define_class("DataTypeError", rb_eStandardError);

  /*
   * Exception raised when the matrix shape is not appropriate for a given operation.
   */
  ShapeError = rb_define_class("ShapeError", rb_eStandardError);

  /*
   * SparseNMatrix Class definition
   */
  SparseNMatrix = rb_define_class("SparseNMatrix", rb_cObject);

  // Class method
  rb_define_alloc_func(SparseNMatrix, nm_sparse_alloc);

  // Singleton Methods
  rb_define_singleton_method(SparseNMatrix, "coo", coo_sparse_nmatrix_init, -1);
  rb_define_singleton_method(SparseNMatrix, "csr", csr_sparse_nmatrix_init, -1);
  rb_define_singleton_method(SparseNMatrix, "csc", csc_sparse_nmatrix_init, -1);
  rb_define_singleton_method(SparseNMatrix, "dia", dia_sparse_nmatrix_init, -1);

  // Instance Methods
  rb_define_method(SparseNMatrix, "dtype",      nm_sparse_get_dtype, 0);
  rb_define_method(SparseNMatrix, "shape",      nm_sparse_get_shape, 0);
  rb_define_method(SparseNMatrix, "to_array",   nm_sparse_to_array, 0);
  rb_define_method(SparseNMatrix, "to_nmatrix", nm_sparse_to_nmatrix, 0);

  /*
   * NMatrix Class definition
   */
  NMatrix = rb_define_class("NMatrix", rb_cObject);

  // Class method
  rb_define_alloc_func(NMatrix, nm_alloc);

  // Instance Methods
  rb_define_method(NMatrix, "initialize", nmatrix_init, -1);
  rb_define_method(NMatrix, "dim",      nm_get_dim, 0);
  rb_define_method(NMatrix, "shape",    nm_get_shape, 0);
  rb_define_method(NMatrix, "elements", nm_get_elements, 0);
  rb_define_method(NMatrix, "dtype",    nm_get_dtype, 0);
  rb_define_method(NMatrix, "stype",    nm_get_stype, 0);

  // Iterators Methods
  rb_define_method(NMatrix, "each", nm_each, 0);
  rb_define_method(NMatrix, "each_with_indices", nm_each_with_indices, 0);
  //rb_define_method(NMatrix, "each_stored_with_indices", nm_each_stored_with_indices, 0);
  //rb_define_method(NMatrix, "map_stored", nm_map_stored, 0);
  //rb_define_method(NMatrix, "each_ordered_stored_with_indices", nm_each_ordered_stored_with_indices, 0);
  rb_define_method(NMatrix, "each_rank", nm_each_rank, 1);
  rb_define_method(NMatrix, "each_row", nm_each_row, 0);
  rb_define_method(NMatrix, "each_column", nm_each_column, 0);
  rb_define_method(NMatrix, "each_layer", nm_each_layer, 0);

  //rb_define_method(NMatrix, "row", nm_get_row, 1);
  //rb_define_method(NMatrix, "column", nm_get_column, 1);

  rb_define_method(NMatrix, "==", nm_eqeq, 1);
  rb_define_method(NMatrix, ">",  nm_gt,   1);
  rb_define_method(NMatrix, ">=", nm_gteq, 1);
  rb_define_method(NMatrix, "<",  nm_lt,   1);
  rb_define_method(NMatrix, "<=", nm_lteq, 1);

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
  rb_define_method(NMatrix, "norm", nm_norm2, 0);
  rb_define_method(NMatrix, "invert", nm_invert, 0);
  rb_define_method(NMatrix, "solve", nm_solve, 1);
  rb_define_method(NMatrix, "det", nm_det, 0);
  rb_define_method(NMatrix, "least_square", nm_least_square, 1);
  rb_define_method(NMatrix, "pinv", nm_pinv, 0);
  rb_define_method(NMatrix, "kronecker_prod", nm_kronecker_prod, 0);
  rb_define_method(NMatrix, "eig", nm_eig, 0);
  rb_define_method(NMatrix, "eigh", nm_eigh, 0);
  rb_define_method(NMatrix, "eigvalsh", nm_eigvalsh, 0);
  rb_define_method(NMatrix, "lu", nm_lu, 0);
  rb_define_method(NMatrix, "lu_factor", nm_lu_factor, 0);
  rb_define_method(NMatrix, "lu_solve", nm_lu_solve, 1);
  rb_define_method(NMatrix, "svd", nm_svd, 0);
  rb_define_method(NMatrix, "svdvals", nm_svdvals, 0);
  rb_define_method(NMatrix, "diagsvd", nm_diagsvd, 0);
  rb_define_method(NMatrix, "orth", nm_orth, 0);
  rb_define_method(NMatrix, "cholesky", nm_cholesky, 0);
  rb_define_method(NMatrix, "cholesky_solve", nm_cholesky_solve, 0);

  rb_define_method(NMatrix, "[]", nm_accessor_get, -1);
  rb_define_method(NMatrix, "[]=", nm_accessor_set, -1);
  rb_define_method(NMatrix, "rank", nm_get_rank, 1);
  rb_define_method(NMatrix, "dtype", nm_get_dtype, 0);
  // rb_define_method(NMatrix, "inspect", nm_inspect, 0);
}

// Return a matrix with all elements value equal to 0
VALUE zeros_nmatrix(int argc, VALUE* argv){
  return constant_nmatrix(argc, argv, 0);
}

// Return a matrix with all elements value equal to 1
VALUE ones_nmatrix(int argc, VALUE* argv){
  return constant_nmatrix(argc, argv, 1);
}

/*
 * Helper function used by 'zeros_nmatrix' and 'ones_nmatrix'
 * to return a matrix with all elements equal to constant
 */
VALUE constant_nmatrix(int argc, VALUE* argv, double constant){
  nmatrix* mat = ALLOC(nmatrix);
  mat->stype = nm_dense;
  mat->dtype = nm_float64;
  mat->ndims = (size_t)RARRAY_LEN(argv[0]);
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

/*
 * Creates a new NMatrix object
 *
 * call-seq:
 *     new shape, initial_array -> NMatrix
 *     new shape, initial_array, dtype -> NMatrix
 *     new shape, initial_array, dtype, stype -> NMatrix
 *     .... TODO: remaining call-sequences
 *
 *     Default value of dtype -> nm_float64
 *     Default value of stype -> nm_dense
 *
 *     shape and initial_array are mendatory.
 *
 *     shape is an array with length equal to number of dimensions. Each value of array is a positive integer
 *     and specifies length of each dimension.
 *
 *     initial_array is an array with length equal to number of elements in the matrix
 *     and specifies initial values of matrix elements.
 */
VALUE nmatrix_init(int argc, VALUE* argv, VALUE self){
  nmatrix* mat;
  Data_Get_Struct(self, nmatrix, mat);

  if(argc > 0){
    mat->ndims = (size_t)RARRAY_LEN(argv[0]);
    mat->count = 1;
    mat->shape = ALLOC_N(size_t, mat->ndims);
    for (size_t index = 0; index < mat->ndims; index++) {
      mat->shape[index] = (size_t)FIX2LONG(RARRAY_AREF(argv[0], index));
      mat->count *= mat->shape[index];
    }
    if(argc < 5){
      mat->stype = (argc > 3) ? nm_stype_from_rbsymbol(argv[3]) : nm_dense;
      mat->dtype = (argc > 2) ? nm_dtype_from_rbsymbol(argv[2]) : nm_float64;
    }
    else{
      mat->stype = (argc > 4) ? nm_stype_from_rbsymbol(argv[5]) : nm_dense;
      mat->dtype = (argc > 5) ? nm_dtype_from_rbsymbol(argv[4]) : nm_float64;
    }

    // Convert and fill the elements values into the NMatrix object
    switch(mat->stype){
      case nm_dense:
      {
        switch(mat->dtype) {
          case nm_bool:
          {
            bool* elements = ALLOC_N(bool, mat->count);
            for (size_t index = 0; index < mat->count; index++) {
              elements[index] = (bool)RTEST(RARRAY_AREF(argv[1], index));
            }
            mat->elements = elements;
            break;
          }
          case nm_int:
          {
            int* elements = ALLOC_N(int, mat->count);
            for (size_t index = 0; index < mat->count; index++) {
              elements[index] = (int)NUM2INT(RARRAY_AREF(argv[1], index));
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
          case nm_float64:
          {
            double* elements = ALLOC_N(double, mat->count);
            for (size_t index = 0; index < mat->count; index++) {
              elements[index] = (double)NUM2DBL(RARRAY_AREF(argv[1], index));
            }
            mat->elements = elements;
            break;
          }
          case nm_complex32:
          {
            float complex* elements = ALLOC_N(float complex, mat->count);
            for (size_t index = 0; index < mat->count; index++) {
              VALUE z = RARRAY_AREF(argv[1], index);
              elements[index] = CMPLXF(NUM2DBL(rb_funcall(z, rb_intern("real"), 0, Qnil)), NUM2DBL(rb_funcall(z, rb_intern("imaginary"), 0, Qnil)));
            }
            mat->elements = elements;
            break;
          }
          case nm_complex64:
          {
            double complex* elements = ALLOC_N(double complex, mat->count);
            for (size_t index = 0; index < mat->count; index++) {
              VALUE z = RARRAY_AREF(argv[1], index);
              elements[index] = CMPLX(NUM2DBL(rb_funcall(z, rb_intern("real"), 0, Qnil)), NUM2DBL(rb_funcall(z, rb_intern("imaginary"), 0, Qnil)));
            }
            mat->elements = elements;
            break;
          }
        }
        break;
      }
      case nm_sparse:
      {
        switch(mat->dtype){
          case nm_float64:
          {
            double* elements = ALLOC_N(double, (size_t)RARRAY_LEN(argv[1]));
            for (size_t index = 0; index < (size_t)RARRAY_LEN(argv[1]); index++) {
              elements[index] = (double)NUM2DBL(RARRAY_AREF(argv[1], index));
            }
            mat->sp = ALLOC(sparse_storage);
            mat->sp->csr = ALLOC(csr_nmatrix);
            mat->sp->csr->count = (size_t)RARRAY_LEN(argv[1]);
            mat->sp->csr->elements = elements;
            mat->sp->csr->ia = ALLOC_N(size_t, (size_t)RARRAY_LEN(argv[2]));
            for (size_t index = 0; index < (size_t)RARRAY_LEN(argv[2]); index++) {
              mat->sp->csr->ia[index] = (size_t)NUM2ULL(RARRAY_AREF(argv[2], index));
            }
            mat->sp->csr->ja = ALLOC_N(size_t, (size_t)RARRAY_LEN(argv[3]));
            for (size_t index = 0; index < (size_t)RARRAY_LEN(argv[3]); index++) {
              mat->sp->csr->ja[index] = (size_t)NUM2ULL(RARRAY_AREF(argv[3], index));
            }
            break;
          }
        }
        break;
      }
    }
  }

  return self;
}

/*
 * Allocator.
 */
VALUE nm_alloc(VALUE klass)
{
  nmatrix* mat = ALLOC(nmatrix);

  return Data_Wrap_Struct(klass, NULL, nm_free, mat);
}

/*
 * Destructor.
 */
void nm_free(nmatrix* mat){
  xfree(mat);
}

// Returns number of dimensions of matrix
VALUE nm_get_dim(VALUE self){
  nmatrix* input;

  Data_Get_Struct(self, nmatrix, input);

  return INT2NUM(input->ndims);
}

// Returns a flat list(one dimensional array) of elements values of matrix
VALUE nm_get_elements(VALUE self){
  nmatrix* input;
  Data_Get_Struct(self, nmatrix, input);

  size_t count = input->count;
  VALUE* array = NULL;

  switch(input->stype){
    case nm_dense:
    {
      array = ALLOC_N(VALUE, input->count);
      switch (input->dtype) {
        case nm_bool:
        {
          bool* elements = (bool*)input->elements;
          for (size_t index = 0; index < input->count; index++){
            array[index] = elements[index] ? Qtrue : Qfalse;
          }
          break;
        }
        case nm_int:
        {
          int* elements = (int*)input->elements;
          for (size_t index = 0; index < input->count; index++){
            array[index] = INT2NUM(elements[index]);
          }
          break;
        }
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
        case nm_complex32:
        {
          float complex* elements = (float complex*)input->elements;
          for (size_t index = 0; index < input->count; index++){
            array[index] = rb_complex_new(DBL2NUM(creal(elements[index])), DBL2NUM(cimag(elements[index])));
          }
          break;
        }
        case nm_complex64:
        {
          double complex* elements = (double complex*)input->elements;
          for (size_t index = 0; index < input->count; index++){
            array[index] = rb_complex_new(DBL2NUM(creal(elements[index])), DBL2NUM(cimag(elements[index])));
          }
          break;
        }
      }
      break;
    }
    case nm_sparse:
    {
      switch(input->dtype){
        case nm_float64:
        {
          count = input->sp->csr->count;
          array = ALLOC_N(VALUE, count);
          double* elements = (double*)input->sp->csr->elements;
          for (size_t index = 0; index < count; index++){
            array[index] = DBL2NUM(elements[index]);
          }
          break;
        }
      }
      break;
    }
  }

  return rb_ary_new4(count, array);
}

/*
 * call-seq:
 *     shape -> Array
 *
 * Get the shape of a matrix, e.g., [2,2]
 */
VALUE nm_get_shape(VALUE self){
  nmatrix* input;

  Data_Get_Struct(self, nmatrix, input);

  VALUE* array = ALLOC_N(VALUE, input->ndims);
  for (size_t index = 0; index < input->ndims; index++){
    array[index] = LONG2NUM(input->shape[index]);
  }

  return rb_ary_new4(input->ndims, array);
}

/*
 * call-seq:
 *     dtype -> Symbol
 *
 * Get the data type (dtype) of a matrix, e.g., :nm_bool, :nm_int,
 * :nm_float32, :nm_float64, :nm_complex32, :nm_complex64
 */
VALUE nm_get_dtype(VALUE self){
  nmatrix* nmat;
  Data_Get_Struct(self, nmatrix, nmat);

  return ID2SYM(rb_intern(DTYPE_NAMES[nmat->dtype]));
}

/*
 * call-seq:
 *     stype -> Symbol
 *
 * Get the storage type (stype) of a matrix, e.g., :nm_sparse, :nm_dense
 */
VALUE nm_get_stype(VALUE self){
  nmatrix* nmat;
  Data_Get_Struct(self, nmatrix, nmat);

  return ID2SYM(rb_intern(STYPE_NAMES[nmat->stype]));
}

VALUE nm_each(VALUE self) {
  nmatrix* input;
  Data_Get_Struct(self, nmatrix, input);

  switch(input->stype){
    case nm_dense:
    {
      switch (input->dtype) {
        case nm_bool:
        {
          bool* elements = (bool*)input->elements;
          for (size_t index = 0; index < input->count; index++){
            rb_yield(elements[index] ? Qtrue : Qfalse);
          }
          break;
        }
        case nm_int:
        {
          int* elements = (int*)input->elements;
          for (size_t index = 0; index < input->count; index++){
            rb_yield(INT2NUM(elements[index]));
          }
          break;
        }
        case nm_float64:
        {
          double* elements = (double*)input->elements;
          for (size_t index = 0; index < input->count; index++){
            rb_yield(DBL2NUM(elements[index]));
          }
          break;
        }
        case nm_float32:
        {
          float* elements = (float*)input->elements;
          for (size_t index = 0; index < input->count; index++){
            rb_yield(DBL2NUM(elements[index]));
          }
          break;
        }
        case nm_complex32:
        {
          float complex* elements = (float complex*)input->elements;
          for (size_t index = 0; index < input->count; index++){
            rb_yield(rb_complex_new(DBL2NUM(creal(elements[index])), DBL2NUM(cimag(elements[index]))));
          }
          break;
        }
        case nm_complex64:
        {
          double complex* elements = (double complex*)input->elements;
          for (size_t index = 0; index < input->count; index++){
            rb_yield(rb_complex_new(DBL2NUM(creal(elements[index])), DBL2NUM(cimag(elements[index]))));
          }
          break;
        }
      }
      break;
    }
    case nm_sparse: //this is to be modified later during sparse work
    {
      switch(input->dtype){
        case nm_float64:
        {
          double* elements = (double*)input->sp->csr->elements;
          for (size_t index = 0; input->sp->csr->count; index++){
            rb_yield(DBL2NUM(elements[index]));
          }
          break;
        }
      }
      break;
    }
  }

  return self;
}

void increment_state(VALUE* state_array, VALUE* shape_array, size_t ndims) {

  for (size_t index = ndims; index > 0; index--) {
    int curr_dim_index = (int)NUM2INT(state_array[index]);
    int curr_dim_length = (int)NUM2INT(shape_array[index - 1]);

    if (curr_dim_index + 1 == curr_dim_length) {
      curr_dim_index = 0;
      state_array[index] = INT2NUM(curr_dim_index);
    } else {
      curr_dim_index++;
      state_array[index] = INT2NUM(curr_dim_index);
      break;
    }

  }
}

VALUE nm_each_with_indices(VALUE self) {
  nmatrix* input;
  Data_Get_Struct(self, nmatrix, input);

  VALUE* shape_array = ALLOC_N(VALUE, input->ndims);
  for (size_t index = 0; index < input->ndims; index++){
    shape_array[index] = LONG2NUM(input->shape[index]);
  }


  //state_array will store the value and indices
  //of current element during iteration
  VALUE* state_array = ALLOC_N(VALUE, input->ndims + 1);
  state_array[0] = -1;  //initialized below inside switch
  for (size_t index = 1; index < input->ndims + 1; index++){
    state_array[index] = INT2NUM(0);
  }

  switch(input->stype){
    case nm_dense:
    {
      switch (input->dtype) {
        case nm_bool:
        {
          bool* elements = (bool*)input->elements;
          for (size_t index = 0; index < input->count; index++){

            state_array[0] = (elements[index] ? Qtrue : Qfalse);

            rb_yield(rb_ary_new4(input->ndims + 1, state_array));

            increment_state(state_array, shape_array, input->ndims);
          }
          break;
        }
        case nm_int:
        {
          int* elements = (int*)input->elements;
          for (size_t index = 0; index < input->count; index++){

            state_array[0] = INT2NUM(elements[index]);

            rb_yield(rb_ary_new4(input->ndims + 1, state_array));

            increment_state(state_array, shape_array, input->ndims);
          }
          break;
        }
        case nm_float64:
        {
          double* elements = (double*)input->elements;
          for (size_t index = 0; index < input->count; index++){

            state_array[0] = DBL2NUM(elements[index]);

            rb_yield(rb_ary_new4(input->ndims + 1, state_array));

            increment_state(state_array, shape_array, input->ndims);
          }
          break;
        }
        case nm_float32:
        {
          float* elements = (float*)input->elements;
          for (size_t index = 0; index < input->count; index++){

            state_array[0] = DBL2NUM(elements[index]);

            rb_yield(rb_ary_new4(input->ndims + 1, state_array));

            increment_state(state_array, shape_array, input->ndims);
          }
          break;
        }
        case nm_complex32:
        {
          float complex* elements = (float complex*)input->elements;
          for (size_t index = 0; index < input->count; index++){

            state_array[0] = rb_complex_new(DBL2NUM(creal(elements[index])), DBL2NUM(cimag(elements[index])));

            rb_yield(rb_ary_new4(input->ndims + 1, state_array));

            increment_state(state_array, shape_array, input->ndims);
          }
          break;
        }
        case nm_complex64:
        {
          double complex* elements = (double complex*)input->elements;
          for (size_t index = 0; index < input->count; index++){

            state_array[0] = rb_complex_new(DBL2NUM(creal(elements[index])), DBL2NUM(cimag(elements[index])));

            rb_yield(rb_ary_new4(input->ndims + 1, state_array));

            increment_state(state_array, shape_array, input->ndims);
          }
          break;
        }
      }
      break;
    }
    case nm_sparse: //this is to be modified later during sparse work
    {
      switch(input->dtype){
        case nm_float64:
        {
          double* elements = (double*)input->sp->csr->elements;
          for (size_t index = 0; index < input->sp->csr->count; index++){

            state_array[0] = DBL2NUM(elements[index]);

            rb_yield(rb_ary_new4(input->ndims + 1, state_array));

            increment_state(state_array, shape_array, input->ndims);
          }
          break;
        }
      }
      break;
    }
  }

  return self;
}

VALUE nm_each_stored_with_indices(VALUE self) {
  return Qnil;
}

VALUE nm_each_ordered_stored_with_indices(VALUE self) {
  return Qnil;
}

VALUE nm_map_stored(VALUE self) {
  return Qnil;
}

VALUE nm_each_rank(VALUE self, VALUE dimension_idx) {
  nmatrix* input;
  Data_Get_Struct(self, nmatrix, input);

  size_t dim_idx = NUM2SIZET(dimension_idx);

  nmatrix* result = ALLOC(nmatrix);
  result->dtype = input->dtype;
  result->stype = input->stype;
  result->count = (input->count / input->shape[dim_idx]);
  result->ndims = (input->ndims) - 1;
  result->shape = ALLOC_N(size_t, result->ndims);

  for(size_t i = 0; i < result->ndims; ++i) {
    if(i < dim_idx)
      result->shape[i] = input->shape[i];
    else
      result->shape[i] = input->shape[i + 1];
  }

  size_t* lower_indices = ALLOC_N(size_t, input->ndims);
  size_t* upper_indices = ALLOC_N(size_t, input->ndims);

  for(size_t i = 0; i < input->ndims; ++i) {
    lower_indices[i] = 0;
    upper_indices[i] = input->shape[i] - 1;
  }
  lower_indices[dim_idx] = upper_indices[dim_idx] = -1;

  for(size_t i = 0; i < input->shape[dim_idx]; ++i) {
    lower_indices[dim_idx] = upper_indices[dim_idx] = i;

    get_slice(input, lower_indices, upper_indices, result);

    rb_yield(Data_Wrap_Struct(NMatrix, NULL, nm_free, result));
  }

  return self;
}

VALUE nm_each_row(VALUE self) {
  return nm_each_rank(self, SIZET2NUM(0));
}

VALUE nm_each_column(VALUE self) {
  return nm_each_rank(self, SIZET2NUM(1));
}

VALUE nm_each_layer(VALUE self) {
  return nm_each_rank(self, SIZET2NUM(2));
}

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

#include "broadcasting.c"

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

/*
 *  Calculates stride using matrix shape array.
 *  Used by get_index to calculate index in flat list of elements
 */

void get_stride(nmatrix* nmat, size_t* stride){
  size_t val = 1;
  for(int i = (nmat->ndims)-1; i >= 0; --i){ //using int here instead of size_t
    stride[i] = val;                         //because size_t does not support
    val *= nmat->shape[i];                    //decrement operator
  }
} 

/*
 *  Calculates index in flat list of elements (stored in backend)
 *  from the given comma separated indices
 */

size_t get_index(nmatrix* nmat, VALUE* indices){
  size_t index = 0;
  size_t* stride = ALLOC_N(size_t, nmat->ndims);
  get_stride(nmat, stride);
  for(size_t i = 0; i < nmat->ndims; ++i){

    if((size_t)FIX2LONG(indices[i]) >= nmat->shape[i] ||
          (int)FIX2LONG(indices[i]) < 0) {  //index out of bounds
      rb_raise(rb_eIndexError, "IndexError: index is out of bounds.");
    }

    index += ((size_t)FIX2LONG(indices[i]) * stride[i]);
  }
  return index;
}

/*
 * converts Range objects to corresponding
 * lower limt and upper limit and put them in size_t varibles
 */
void parse_ranges(nmatrix* nmat, VALUE* indices, size_t* lower, size_t* upper){

  for(size_t i = 0; i < nmat->ndims; ++i) {
    //take each indices value and parse it
    //to get the corr start and end index of the range

    size_t a1, b1;

    if(rb_obj_is_kind_of(indices[i], rb_cRange) == Qtrue){

      VALUE range_begin = rb_funcall(indices[i], rb_intern("begin"), 0);
      VALUE range_end = rb_funcall(indices[i], rb_intern("end"), 0);
      VALUE exclude_end = rb_funcall(indices[i], rb_intern("exclude_end?"), 0);

      a1 = NUM2SIZET(range_begin);
      if(range_end == Qnil){ //end-less range
        //assign (size_of_dim - 1) to b1
        b1 = NUM2SIZET(nmat->shape[i]) - 1;
      }
      else{
        b1 = NUM2SIZET(range_end);
      }
      

      if(exclude_end == Qtrue && range_end != Qnil){
        b1--;
      }

    }
    else{
      a1 = b1 = NUM2SIZET(indices[i]);
    }

    if(a1 > b1){
      //raise invalid range error
    }

    if(a1 >= nmat->shape[i] || b1 >= nmat->shape[i]){
      //raise index out of bounds error
    }

    lower[i] = a1, upper[i] = b1;
  }

}

/*
 *
 *
 */
void get_slice(nmatrix* nmat, size_t* lower, size_t* upper, nmatrix* slice){
  /*
    parse the indices to form ranges for C loops

    then use them to fill up the elements
  */

  size_t slice_count = 1, slice_ndims = 0;

  for(size_t i = 0; i < nmat->ndims; ++i){
    size_t a1 = lower[i], b1 = upper[i];

    //if range len is > 1, then inc slice_ndims by 1
    //and slice_count would be prod of all ranges len
    if(b1 - a1 > 0){
      slice_ndims++;
      slice_count *= (b1 - a1 + 1);
    }
  }

  slice->count = slice_count;
  slice->ndims = slice_ndims;
  slice->shape = ALLOC_N(size_t, slice->ndims);

  size_t slice_ind = 0;
  for(size_t i = 0; i < nmat->ndims; ++i){
    size_t dim_length = (upper[i] - lower[i] + 1);
    if(dim_length == 1)
      continue;
    slice->shape[slice_ind++] = dim_length;
  }


  //mark elements that are inside the slice
  //and copy them to elements array of slice

  VALUE* state_array = ALLOC_N(VALUE, nmat->ndims);
  for(size_t i = 0; i < nmat->ndims; ++i){
    state_array[i] = SIZET2NUM(lower[i]);
  }

  switch (nmat->dtype){
    case nm_bool:
    {
      bool* nmat_elements = (bool*)nmat->elements;

      bool* slice_elements = ALLOC_N(bool, slice->count);

      for(size_t i = 0; i < slice->count; ++i){
        size_t nmat_index = get_index(nmat, state_array);
        slice_elements[i] = nmat_elements[nmat_index];

        size_t state_index = (nmat->ndims) - 1;
        while(true){
          size_t curr_index_value = NUM2SIZET(state_array[state_index]);

          if(curr_index_value == upper[state_index]){
            curr_index_value = lower[state_index];
            state_array[state_index] = SIZET2NUM(curr_index_value);
          }
          else{
            curr_index_value++;
            state_array[state_index] = SIZET2NUM(curr_index_value);
            break;
          }  

          state_index--;        
        }
      }

      slice->elements = slice_elements;
      break;
    }
    case nm_int:
    {
      int* nmat_elements = (int*)nmat->elements;

      int* slice_elements = ALLOC_N(int, slice->count);

      for(size_t i = 0; i < slice->count; ++i){
        size_t nmat_index = get_index(nmat, state_array);
        slice_elements[i] = nmat_elements[nmat_index];

        size_t state_index = (nmat->ndims) - 1;
        while(true){
          size_t curr_index_value = NUM2SIZET(state_array[state_index]);

          if(curr_index_value == upper[state_index]){
            curr_index_value = lower[state_index];
            state_array[state_index] = SIZET2NUM(curr_index_value);
          }
          else{
            curr_index_value++;
            state_array[state_index] = SIZET2NUM(curr_index_value);
            break;
          }  

          state_index--;        
        }
      }

      slice->elements = slice_elements;
      break;
    }
    case nm_float64:
    {
      double* nmat_elements = (double*)nmat->elements;

      double* slice_elements = ALLOC_N(double, slice->count);

      for(size_t i = 0; i < slice->count; ++i){
        size_t nmat_index = get_index(nmat, state_array);
        slice_elements[i] = nmat_elements[nmat_index];

        size_t state_index = (nmat->ndims) - 1;
        while(true){
          size_t curr_index_value = NUM2SIZET(state_array[state_index]);

          if(curr_index_value == upper[state_index]){
            curr_index_value = lower[state_index];
            state_array[state_index] = SIZET2NUM(curr_index_value);
          }
          else{
            curr_index_value++;
            state_array[state_index] = SIZET2NUM(curr_index_value);
            break;
          }  

          state_index--;        
        }
      }

      slice->elements = slice_elements;
      break;
    }
    case nm_float32:
    {
      float* nmat_elements = (float*)nmat->elements;

      float* slice_elements = ALLOC_N(float, slice->count);

      for(size_t i = 0; i < slice->count; ++i){
        size_t nmat_index = get_index(nmat, state_array);
        slice_elements[i] = nmat_elements[nmat_index];

        size_t state_index = (nmat->ndims) - 1;
        while(true){
          size_t curr_index_value = NUM2SIZET(state_array[state_index]);

          if(curr_index_value == upper[state_index]){
            curr_index_value = lower[state_index];
            state_array[state_index] = SIZET2NUM(curr_index_value);
          }
          else{
            curr_index_value++;
            state_array[state_index] = SIZET2NUM(curr_index_value);
            break;
          }  

          state_index--;        
        }
      }

      slice->elements = slice_elements;
      break;
    }
    case nm_complex32:
    {
      float complex* nmat_elements = (float complex*)nmat->elements;

      float complex* slice_elements = ALLOC_N(float complex, slice->count);

      for(size_t i = 0; i < slice->count; ++i){
        size_t nmat_index = get_index(nmat, state_array);
        slice_elements[i] = nmat_elements[nmat_index];

        size_t state_index = (nmat->ndims) - 1;
        while(true){
          size_t curr_index_value = NUM2SIZET(state_array[state_index]);

          if(curr_index_value == upper[state_index]){
            curr_index_value = lower[state_index];
            state_array[state_index] = SIZET2NUM(curr_index_value);
          }
          else{
            curr_index_value++;
            state_array[state_index] = SIZET2NUM(curr_index_value);
            break;
          }  

          state_index--;        
        }
      }

      slice->elements = slice_elements;
      break;
    }
    case nm_complex64:
    {
      double complex* nmat_elements = (double complex*)nmat->elements;

      double complex* slice_elements = ALLOC_N(double complex, slice->count);

      for(size_t i = 0; i < slice->count; ++i){
        size_t nmat_index = get_index(nmat, state_array);
        slice_elements[i] = nmat_elements[nmat_index];

        size_t state_index = (nmat->ndims) - 1;
        while(true){
          size_t curr_index_value = NUM2SIZET(state_array[state_index]);

          if(curr_index_value == upper[state_index]){
            curr_index_value = lower[state_index];
            state_array[state_index] = SIZET2NUM(curr_index_value);
          }
          else{
            curr_index_value++;
            state_array[state_index] = SIZET2NUM(curr_index_value);
            break;
          }  

          state_index--;        
        }
      }

      slice->elements = slice_elements;
      break;
    }
  }

  //fill the nmatrix* slice with the req data
}

/*
 *  checks if the given set of indices corresponds
 *  to a single element or a slice.
 *
 */
bool is_slice(nmatrix* nmat, VALUE* indices){
  for(size_t i = 0; i < nmat->ndims; ++i){
    if(rb_obj_is_kind_of(indices[i], rb_cRange) == Qtrue)
      return true;
  }

  return false;
}

/*
 * Get the element of a matrix at the given index
 */
VALUE nm_accessor_get(int argc, VALUE* argv, VALUE self){
  nmatrix* nmat;
  Data_Get_Struct(self, nmatrix, nmat);

  size_t* lower_indices = ALLOC_N(size_t, nmat->ndims);
  size_t* upper_indices = ALLOC_N(size_t, nmat->ndims);

  parse_ranges(nmat, argv, lower_indices, upper_indices);

  switch(nmat->stype){
    case nm_dense:
    {
      switch (nmat->dtype){
        case nm_bool:
        {
          if(is_slice(nmat, argv)){

            nmatrix* slice = ALLOC(nmatrix);
            slice->dtype = nmat->dtype;
            slice->stype = nmat->stype;

            get_slice(nmat, lower_indices, upper_indices, slice);

            return Data_Wrap_Struct(NMatrix, NULL, nm_free, slice);

            //return a slice
          }
          else{
            size_t index = get_index(nmat, argv);

            bool* elements = (bool*)nmat->elements;
            bool val = elements[index];
            return (val ? Qtrue : Qfalse);
          }
          
          break;
        }
        case nm_int:
        {
          if(is_slice(nmat, argv)){

            nmatrix* slice = ALLOC(nmatrix);
            slice->dtype = nmat->dtype;
            slice->stype = nmat->stype;

            get_slice(nmat, lower_indices, upper_indices, slice);

            return Data_Wrap_Struct(NMatrix, NULL, nm_free, slice);

            //return a slice
          }
          else{
            size_t index = get_index(nmat, argv);

            int* elements = (int*)nmat->elements;
            int val = elements[index];
            return INT2NUM(val);
          }
          
          break;
        }
        case nm_float64:
        {
          if(is_slice(nmat, argv)){

            nmatrix* slice = ALLOC(nmatrix);
            slice->dtype = nmat->dtype;
            slice->stype = nmat->stype;

            get_slice(nmat, lower_indices, upper_indices, slice);

            return Data_Wrap_Struct(NMatrix, NULL, nm_free, slice);

            //return a slice
          }
          else{
            size_t index = get_index(nmat, argv);
          
            double* elements = (double*)nmat->elements;
            double val = elements[index];
            return DBL2NUM(val);
          }

          break;
        }
        case nm_float32:
        {
          if(is_slice(nmat, argv)){

            nmatrix* slice = ALLOC(nmatrix);
            slice->dtype = nmat->dtype;
            slice->stype = nmat->stype;

            get_slice(nmat, lower_indices, upper_indices, slice);

            return Data_Wrap_Struct(NMatrix, NULL, nm_free, slice);

            //return a slice
          }
          else{
            size_t index = get_index(nmat, argv);

            float* elements = (float*)nmat->elements;
            float val = elements[index];
            return DBL2NUM(val);
          }
          
          break;
        }
        case nm_complex32:
        {
          if(is_slice(nmat, argv)){

            nmatrix* slice = ALLOC(nmatrix);
            slice->dtype = nmat->dtype;
            slice->stype = nmat->stype;

            get_slice(nmat, lower_indices, upper_indices, slice);

            return Data_Wrap_Struct(NMatrix, NULL, nm_free, slice);

            //return a slice
          }
          else{
            size_t index = get_index(nmat, argv);

            float complex* elements = (float complex*)nmat->elements;
            float complex val = elements[index];
            return rb_complex_new(DBL2NUM(creal(val)), DBL2NUM(cimag(val)));
          }
          
          break;
        }
        case nm_complex64:
        {
          if(is_slice(nmat, argv)){

            nmatrix* slice = ALLOC(nmatrix);
            slice->dtype = nmat->dtype;
            slice->stype = nmat->stype;

            get_slice(nmat, lower_indices, upper_indices, slice);

            return Data_Wrap_Struct(NMatrix, NULL, nm_free, slice);

            //return a slice
          }
          else{
            size_t index = get_index(nmat, argv);

            double complex* elements = (double complex*)nmat->elements;
            double complex val = elements[index];
            return rb_complex_new(DBL2NUM(creal(val)), DBL2NUM(cimag(val)));
          }
          
          break;
        }
      }
      break;
    }
    case nm_sparse: //this is to be modified later during sparse work
    {
      switch(nmat->dtype){
        case nm_float64:
        {
          if(is_slice(nmat, argv)){
            //raise not implemented error
          }
          else{
            size_t index = get_index(nmat, argv);

            double* elements = (double*)nmat->sp->csr->elements;
            double val = elements[index];
            return DBL2NUM(val);
            
            break;
          }
        }
      }
      break;
    }
  }

  return DBL2NUM(-1);   //make it more descriptive

}

/*
 * Change the value of element at given index of a matrix to the given value
 */
VALUE nm_accessor_set(int argc, VALUE* argv, VALUE self){
  nmatrix* nmat;
  Data_Get_Struct(self, nmatrix, nmat);

  size_t index = get_index(nmat, argv);

  switch(nmat->stype){
    case nm_dense:
    {
      switch (nmat->dtype){
        case nm_bool:
        {
          bool* elements = (bool*)nmat->elements;
          elements[index] = RTEST(argv[nmat->ndims]);
          nmat->elements = elements;
          return argv[2];
          
          break;
        }
        case nm_int:
        {
          int* elements = (int*)nmat->elements;
          elements[index] = NUM2DBL(argv[nmat->ndims]);
          nmat->elements = elements;
          return argv[2];
          
          break;
        }
        case nm_float64:
        {
          double* elements = (double*)nmat->elements;
          elements[index] = NUM2DBL(argv[nmat->ndims]);

          nmat->elements = elements;
          return argv[2];
        }
        case nm_float32:
        {
          float* elements = (float*)nmat->elements;
          elements[index] = NUM2DBL(argv[nmat->ndims]);

          nmat->elements = elements;
          return argv[2];
          
          break;
        }
        case nm_complex32:
        {
          float complex* elements = (float complex*)nmat->elements;
          elements[index] = NUM2DBL(argv[nmat->ndims]);

          nmat->elements = elements;
          return argv[2];
          
          break;
        }
        case nm_complex64:
        {
          double complex* elements = (double complex*)nmat->elements;
          elements[index] = NUM2DBL(argv[nmat->ndims]);

          nmat->elements = elements;
          return argv[2];
          
          break;
        }
      }
      break;
    }
    case nm_sparse: //this is to be modified later during sparse work
    {
      switch(nmat->dtype){
        case nm_float64:
        {
          double* elements = (double*)nmat->sp->csr->elements;
          elements[index] = NUM2DBL(argv[nmat->ndims]);
          nmat->elements = elements;
          return argv[2];
          
          break;
        }
      }
      break;
    }
  }

  return argv[2];
}

// Return rank of the matrix
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
    result_elements[i] = input_elements[input->shape[1]*dim + i];
  result->elements = result_elements;

  return Data_Wrap_Struct(NMatrix, NULL, nm_free, result);
}

VALUE nm_inspect(VALUE self){
  const char* c = "Class: NMatrix";

  return rb_str_new_cstr(c);
}


#include "blas.c"
#include "lapack.c"
#include "sparse.c"
#include "statistics.c"
