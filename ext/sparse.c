VALUE nm_sparse_alloc(VALUE klass)
{
  csr_nmatrix* mat = ALLOC(csr_nmatrix);

  return Data_Wrap_Struct(klass, NULL, nm_free, mat);
}

void nm_sparse_free(csr_nmatrix* mat){
  xfree(mat);
}

VALUE coo_sparse_nmatrix_init(int argc, VALUE* argv){
  sparse_nmatrix* mat = ALLOC(sparse_nmatrix);

  if(argc > 0){
    mat->ndims = 2;
    mat->count = 1;
    mat->shape = ALLOC_N(size_t, mat->ndims);
    for (size_t index = 0; index < mat->ndims; index++) {
      mat->shape[index] = (size_t)FIX2LONG(RARRAY_AREF(argv[0], index));
      mat->count *= mat->shape[index];
    }
    mat->dtype = nm_dtype_from_rbsymbol(argv[4]);
    mat->sptype = coo;
    mat->coo = ALLOC(coo_nmatrix);
    mat->coo->count = (size_t)RARRAY_LEN(argv[1]);
    mat->coo->ia = ALLOC_N(size_t, (size_t)RARRAY_LEN(argv[2]));
    for (size_t index = 0; index < (size_t)RARRAY_LEN(argv[2]); index++) {
      mat->coo->ia[index] = (size_t)NUM2ULL(RARRAY_AREF(argv[2], index));
    }
    mat->coo->ja = ALLOC_N(size_t, (size_t)RARRAY_LEN(argv[3]));
    for (size_t index = 0; index < (size_t)RARRAY_LEN(argv[3]); index++) {
      mat->coo->ja[index] = (size_t)NUM2ULL(RARRAY_AREF(argv[3], index));
    }

    switch(mat->dtype){
      case nm_bool:
      {
        bool* elements = ALLOC_N(bool, (size_t)RARRAY_LEN(argv[1]));
        for (size_t index = 0; index < (size_t)RARRAY_LEN(argv[1]); index++) {
          elements[index] = (bool)NUM2DBL(RARRAY_AREF(argv[1], index));
        }
        mat->coo->elements = elements;
        break;
      }
      case nm_int:
      {
        int* elements = ALLOC_N(int, (size_t)RARRAY_LEN(argv[1]));
        for (size_t index = 0; index < (size_t)RARRAY_LEN(argv[1]); index++) {
          elements[index] = (int)NUM2DBL(RARRAY_AREF(argv[1], index));
        }
        mat->coo->elements = elements;
        break;
      }
      case nm_float32:
      {
        float* elements = ALLOC_N(float, (size_t)RARRAY_LEN(argv[1]));
        for (size_t index = 0; index < (size_t)RARRAY_LEN(argv[1]); index++) {
          elements[index] = (float)NUM2DBL(RARRAY_AREF(argv[1], index));
        }
        mat->coo->elements = elements;
        break;
      }
      case nm_float64:
      {
        double* elements = ALLOC_N(double, (size_t)RARRAY_LEN(argv[1]));
        for (size_t index = 0; index < (size_t)RARRAY_LEN(argv[1]); index++) {
          elements[index] = (double)NUM2DBL(RARRAY_AREF(argv[1], index));
        }
        mat->coo->elements = elements;
        break;
      }
      case nm_complex32:
      {
        float complex* elements = ALLOC_N(float complex, (size_t)RARRAY_LEN(argv[1]));
        for (size_t index = 0; index < (size_t)RARRAY_LEN(argv[1]); index++) {
          VALUE z = RARRAY_AREF(argv[1], index);
          elements[index] = CMPLXF(NUM2DBL(rb_funcall(z, rb_intern("real"), 0, Qnil)), NUM2DBL(rb_funcall(z, rb_intern("imaginary"), 0, Qnil)));
        }
        mat->coo->elements = elements;
        break;
      }
      case nm_complex64:
      {
        float complex* elements = ALLOC_N(float complex, (size_t)RARRAY_LEN(argv[1]));
        for (size_t index = 0; index < (size_t)RARRAY_LEN(argv[1]); index++) {
          VALUE z = RARRAY_AREF(argv[1], index);
          elements[index] = CMPLX(NUM2DBL(rb_funcall(z, rb_intern("real"), 0, Qnil)), NUM2DBL(rb_funcall(z, rb_intern("imaginary"), 0, Qnil)));
        }
        mat->coo->elements = elements;
        break;
      }
    }
  }

  return Data_Wrap_Struct(SparseNMatrix, NULL, nm_free, mat);
}

VALUE csr_sparse_nmatrix_init(int argc, VALUE* argv){
  sparse_nmatrix* mat = ALLOC(sparse_nmatrix);

  if(argc > 0){
    mat->ndims = 2;
    mat->count = 1;
    mat->shape = ALLOC_N(size_t, mat->ndims);
    for (size_t index = 0; index < mat->ndims; index++) {
      mat->shape[index] = (size_t)FIX2LONG(RARRAY_AREF(argv[0], index));
      mat->count *= mat->shape[index];
    }
    mat->dtype = nm_dtype_from_rbsymbol(argv[4]);
    mat->sptype = csr;
    mat->csr = ALLOC(csr_nmatrix);
    mat->csr->count = (size_t)RARRAY_LEN(argv[1]);
    mat->csr->ia = ALLOC_N(size_t, (size_t)RARRAY_LEN(argv[2]));
    for (size_t index = 0; index < (size_t)RARRAY_LEN(argv[2]); index++) {
      mat->csr->ia[index] = (size_t)NUM2ULL(RARRAY_AREF(argv[2], index));
    }
    mat->csr->ja = ALLOC_N(size_t, (size_t)RARRAY_LEN(argv[3]));
    for (size_t index = 0; index < (size_t)RARRAY_LEN(argv[3]); index++) {
      mat->csr->ja[index] = (size_t)NUM2ULL(RARRAY_AREF(argv[3], index));
    }

    switch(mat->dtype){
      case nm_bool:
      {
        bool* elements = ALLOC_N(bool, (size_t)RARRAY_LEN(argv[1]));
        for (size_t index = 0; index < (size_t)RARRAY_LEN(argv[1]); index++) {
          elements[index] = (bool)NUM2DBL(RARRAY_AREF(argv[1], index));
        }
        mat->csr->elements = elements;
        break;
      }
      case nm_int:
      {
        int* elements = ALLOC_N(int, (size_t)RARRAY_LEN(argv[1]));
        for (size_t index = 0; index < (size_t)RARRAY_LEN(argv[1]); index++) {
          elements[index] = (int)NUM2DBL(RARRAY_AREF(argv[1], index));
        }
        mat->csr->elements = elements;
        break;
      }
      case nm_float32:
      {
        float* elements = ALLOC_N(float, (size_t)RARRAY_LEN(argv[1]));
        for (size_t index = 0; index < (size_t)RARRAY_LEN(argv[1]); index++) {
          elements[index] = (float)NUM2DBL(RARRAY_AREF(argv[1], index));
        }
        mat->csr->elements = elements;
        break;
      }
      case nm_float64:
      {
        double* elements = ALLOC_N(double, (size_t)RARRAY_LEN(argv[1]));
        for (size_t index = 0; index < (size_t)RARRAY_LEN(argv[1]); index++) {
          elements[index] = (double)NUM2DBL(RARRAY_AREF(argv[1], index));
        }
        mat->csr->elements = elements;
        break;
      }
      case nm_complex32:
      {
        float complex* elements = ALLOC_N(float complex, (size_t)RARRAY_LEN(argv[1]));
        for (size_t index = 0; index < (size_t)RARRAY_LEN(argv[1]); index++) {
          VALUE z = RARRAY_AREF(argv[1], index);
          elements[index] = CMPLXF(NUM2DBL(rb_funcall(z, rb_intern("real"), 0, Qnil)), NUM2DBL(rb_funcall(z, rb_intern("imaginary"), 0, Qnil)));
        }
        mat->csr->elements = elements;
        break;
      }
      case nm_complex64:
      {
        float complex* elements = ALLOC_N(float complex, (size_t)RARRAY_LEN(argv[1]));
        for (size_t index = 0; index < (size_t)RARRAY_LEN(argv[1]); index++) {
          VALUE z = RARRAY_AREF(argv[1], index);
          elements[index] = CMPLX(NUM2DBL(rb_funcall(z, rb_intern("real"), 0, Qnil)), NUM2DBL(rb_funcall(z, rb_intern("imaginary"), 0, Qnil)));
        }
        mat->csr->elements = elements;
        break;
      }
    }
  }

  return Data_Wrap_Struct(SparseNMatrix, NULL, nm_free, mat);
}

VALUE csc_sparse_nmatrix_init(int argc, VALUE* argv){
  sparse_nmatrix* mat = ALLOC(sparse_nmatrix);

  if(argc > 0){
    mat->ndims = 2;
    mat->count = 1;
    mat->shape = ALLOC_N(size_t, mat->ndims);
    for (size_t index = 0; index < mat->ndims; index++) {
      mat->shape[index] = (size_t)FIX2LONG(RARRAY_AREF(argv[0], index));
      mat->count *= mat->shape[index];
    }
    mat->dtype = nm_dtype_from_rbsymbol(argv[4]);
    mat->sptype = csc;
    mat->csc = ALLOC(csc_nmatrix);
    mat->csc->count = (size_t)RARRAY_LEN(argv[1]);
    mat->csc->ia = ALLOC_N(size_t, (size_t)RARRAY_LEN(argv[2]));
    for (size_t index = 0; index < (size_t)RARRAY_LEN(argv[2]); index++) {
      mat->csc->ia[index] = (size_t)NUM2ULL(RARRAY_AREF(argv[2], index));
    }
    mat->csc->ja = ALLOC_N(size_t, (size_t)RARRAY_LEN(argv[3]));
    for (size_t index = 0; index < (size_t)RARRAY_LEN(argv[3]); index++) {
      mat->csc->ja[index] = (size_t)NUM2ULL(RARRAY_AREF(argv[3], index));
    }

    switch(mat->dtype){
      case nm_bool:
      {
        bool* elements = ALLOC_N(bool, (size_t)RARRAY_LEN(argv[1]));
        for (size_t index = 0; index < (size_t)RARRAY_LEN(argv[1]); index++) {
          elements[index] = (bool)NUM2DBL(RARRAY_AREF(argv[1], index));
        }
        mat->csc->elements = elements;
        break;
      }
      case nm_int:
      {
        int* elements = ALLOC_N(int, (size_t)RARRAY_LEN(argv[1]));
        for (size_t index = 0; index < (size_t)RARRAY_LEN(argv[1]); index++) {
          elements[index] = (int)NUM2DBL(RARRAY_AREF(argv[1], index));
        }
        mat->csc->elements = elements;
        break;
      }
      case nm_float32:
      {
        float* elements = ALLOC_N(float, (size_t)RARRAY_LEN(argv[1]));
        for (size_t index = 0; index < (size_t)RARRAY_LEN(argv[1]); index++) {
          elements[index] = (float)NUM2DBL(RARRAY_AREF(argv[1], index));
        }
        mat->csc->elements = elements;
        break;
      }
      case nm_float64:
      {
        double* elements = ALLOC_N(double, (size_t)RARRAY_LEN(argv[1]));
        for (size_t index = 0; index < (size_t)RARRAY_LEN(argv[1]); index++) {
          elements[index] = (double)NUM2DBL(RARRAY_AREF(argv[1], index));
        }
        mat->csc->elements = elements;
        break;
      }
      case nm_complex32:
      {
        float complex* elements = ALLOC_N(float complex, (size_t)RARRAY_LEN(argv[1]));
        for (size_t index = 0; index < (size_t)RARRAY_LEN(argv[1]); index++) {
          VALUE z = RARRAY_AREF(argv[1], index);
          elements[index] = CMPLXF(NUM2DBL(rb_funcall(z, rb_intern("real"), 0, Qnil)), NUM2DBL(rb_funcall(z, rb_intern("imaginary"), 0, Qnil)));
        }
        mat->csc->elements = elements;
        break;
      }
      case nm_complex64:
      {
        float complex* elements = ALLOC_N(float complex, (size_t)RARRAY_LEN(argv[1]));
        for (size_t index = 0; index < (size_t)RARRAY_LEN(argv[1]); index++) {
          VALUE z = RARRAY_AREF(argv[1], index);
          elements[index] = CMPLX(NUM2DBL(rb_funcall(z, rb_intern("real"), 0, Qnil)), NUM2DBL(rb_funcall(z, rb_intern("imaginary"), 0, Qnil)));
        }
        mat->csc->elements = elements;
        break;
      }
    }
  }

  return Data_Wrap_Struct(SparseNMatrix, NULL, nm_free, mat);
}

VALUE dia_sparse_nmatrix_init(int argc, VALUE* argv){
  sparse_nmatrix* mat = ALLOC(sparse_nmatrix);

  if(argc > 0){
    mat->ndims = 2;
    mat->count = 1;
    mat->shape = ALLOC_N(size_t, mat->ndims);
    for (size_t index = 0; index < mat->ndims; index++) {
      mat->shape[index] = (size_t)FIX2LONG(RARRAY_AREF(argv[0], index));
      mat->count *= mat->shape[index];
    }
    mat->dtype = nm_dtype_from_rbsymbol(argv[3]);
    mat->sptype = dia;
    mat->diag = ALLOC(diag_nmatrix);
    mat->diag->count = (size_t)RARRAY_LEN(argv[1]);
    mat->diag->offset = ALLOC_N(size_t, (size_t)RARRAY_LEN(argv[2]));
    for (size_t index = 0; index < (size_t)RARRAY_LEN(argv[2]); index++) {
      mat->diag->offset[index] = (size_t)NUM2ULL(RARRAY_AREF(argv[2], index));
    }

    switch(mat->dtype){
      case nm_int:
      {
        int* elements = ALLOC_N(int, (size_t)RARRAY_LEN(argv[1]));
        for (size_t index = 0; index < (size_t)RARRAY_LEN(argv[1]); index++) {
          elements[index] = (int)NUM2DBL(RARRAY_AREF(argv[1], index));
        }
        mat->diag->elements = elements;
        break;
      }
      case nm_float32:
      {
        float* elements = ALLOC_N(float, (size_t)RARRAY_LEN(argv[1]));
        for (size_t index = 0; index < (size_t)RARRAY_LEN(argv[1]); index++) {
          elements[index] = (float)NUM2DBL(RARRAY_AREF(argv[1], index));
        }
        mat->diag->elements = elements;
        break;
      }
      case nm_float64:
      {
        double* elements = ALLOC_N(double, (size_t)RARRAY_LEN(argv[1]));
        for (size_t index = 0; index < (size_t)RARRAY_LEN(argv[1]); index++) {
          elements[index] = (double)NUM2DBL(RARRAY_AREF(argv[1], index));
        }
        mat->diag->elements = elements;
        break;
      }
      case nm_complex32:
      {
        float complex* elements = ALLOC_N(float complex, (size_t)RARRAY_LEN(argv[1]));
        for (size_t index = 0; index < (size_t)RARRAY_LEN(argv[1]); index++) {
          VALUE z = RARRAY_AREF(argv[1], index);
          elements[index] = CMPLXF(NUM2DBL(rb_funcall(z, rb_intern("real"), 0, Qnil)), NUM2DBL(rb_funcall(z, rb_intern("imaginary"), 0, Qnil)));
        }
        mat->diag->elements = elements;
        break;
      }
      case nm_complex64:
      {
        float complex* elements = ALLOC_N(float complex, (size_t)RARRAY_LEN(argv[1]));
        for (size_t index = 0; index < (size_t)RARRAY_LEN(argv[1]); index++) {
          VALUE z = RARRAY_AREF(argv[1], index);
          elements[index] = CMPLX(NUM2DBL(rb_funcall(z, rb_intern("real"), 0, Qnil)), NUM2DBL(rb_funcall(z, rb_intern("imaginary"), 0, Qnil)));
        }
        mat->diag->elements = elements;
        break;
      }
    }
  }

  return Data_Wrap_Struct(SparseNMatrix, NULL, nm_free, mat);
}

VALUE nm_sparse_get_dtype(VALUE self){
  sparse_nmatrix* spmat;
  Data_Get_Struct(self, sparse_nmatrix, spmat);

  return rb_str_new_cstr(DTYPE_NAMES[spmat->dtype]);
}

VALUE nm_sparse_get_shape(VALUE self){
  sparse_nmatrix* input;

  Data_Get_Struct(self, sparse_nmatrix, input);

  VALUE* array = ALLOC_N(VALUE, input->ndims);
  for (size_t index = 0; index < input->ndims; index++){
    array[index] = LONG2NUM(input->shape[index]);
  }

  return rb_ary_new4(input->ndims, array);
}

VALUE nm_sparse_to_array(VALUE self){
  sparse_nmatrix* input;
  Data_Get_Struct(self, sparse_nmatrix, input);

  size_t count = input->count;
  VALUE* array = ALLOC_N(VALUE, input->count);
  switch (input->dtype) {
    case coo:
    {
      double* elements = ALLOC_N(double, count);
      get_dense_from_coo(input->coo->elements,
                        input->shape[0],
                        input->shape[1],
                        input->coo->ia,
                        input->coo->ja,
                        elements);
      for (size_t index = 0; index < count; index++){
        array[index] = DBL2NUM(elements[index]);
      }
      break;
    }
    case csc:
    {
      double* elements = ALLOC_N(double, count);
      get_dense_from_csc(input->csc->elements,
                        input->shape[0],
                        input->shape[1],
                        input->csc->ia,
                        input->csc->ja,
                        elements);
      for (size_t index = 0; index < count; index++){
        array[index] = DBL2NUM(elements[index]);
      }
      break;
    }
    case csr:
    {
      double* elements = ALLOC_N(double, count);
      get_dense_from_csr(input->csr->elements,
                        input->shape[0],
                        input->shape[1],
                        input->csr->ia,
                        input->csr->ja,
                        elements);
      for (size_t index = 0; index < count; index++){
        array[index] = DBL2NUM(elements[index]);
      }
      break;
    }
    case dia:
    {
      double* elements = ALLOC_N(double, count);
      get_dense_from_dia(input->diag->elements,
                        input->shape[0],
                        input->shape[1],
                        input->diag->offset,
                        elements);
      for (size_t index = 0; index < count; index++){
        array[index] = DBL2NUM(elements[index]);
      }
      break;
    }
  }
  return rb_ary_new4(count, array);
}

VALUE nm_sparse_to_nmatrix(VALUE self){
  sparse_nmatrix* input;
  Data_Get_Struct(self, sparse_nmatrix, input);

  nmatrix* result = ALLOC(nmatrix);
  result->dtype = input->dtype;
  result->stype = nm_dense;
  result->ndims = input->ndims;
  result->shape = ALLOC_N(size_t, result->ndims);

  result->shape[0] = input->shape[0];
  result->shape[1] = input->shape[1];
  result->count = input->count;

  switch (input->dtype) {
    case nm_float64:
    {
      double* elements = ALLOC_N(double, result->count);
      get_dense_from_csr(input->csr->elements,
                        input->shape[0],
                        input->shape[1],
                        input->csr->ia,
                        input->csr->ja,
                        elements);
      result->elements = elements;
      break;
    }
  }
  return Data_Wrap_Struct(NMatrix, NULL, nm_free, result);
}

void get_dense_from_coo(const double* data, const size_t rows,
                        const size_t cols, const size_t* ia,
                        const size_t* ja, double* elements){
  for(size_t i = 0; i < rows*cols; ++i){ elements[i] = 0; }

  size_t index = 0;

  for(size_t i = 1; i < rows + 1; ++i){
    for(size_t j = 0; j < ia[i] - ia[i - 1]; ++j){
      elements[(i-1)*cols + ja[index]] = data[index];
      index++;
    }
  }
}

void get_dense_from_csc(const double* data, const size_t rows,
                        const size_t cols, const size_t* ia,
                        const size_t* ja, double* elements){
  for(size_t i = 0; i < rows*cols; ++i){ elements[i] = 0; }

  size_t index = 0;

  for(size_t i = 1; i < rows + 1; ++i){
    for(size_t j = 0; j < ia[i] - ia[i - 1]; ++j){
      elements[(i-1)*cols + ja[index]] = data[index];
      index++;
    }
  }
}

void get_dense_from_csr(const double* data, const size_t rows,
                        const size_t cols, const size_t* ia,
                        const size_t* ja, double* elements){
  for(size_t i = 0; i < rows*cols; ++i){ elements[i] = 0; }

  size_t index = 0;

  for(size_t i = 1; i < rows + 1; ++i){
    for(size_t j = 0; j < ia[i] - ia[i - 1]; ++j){
      elements[(i-1)*cols + ja[index]] = data[index];
      index++;
    }
  }
}

void get_dense_from_dia(const double* data, const size_t rows,
                        const size_t cols, const size_t* offset,
                        double* elements){
  for(size_t i = 0; i < rows*cols; ++i){ elements[i] = 0; }

  size_t index = 0;

  for(size_t i = 1; i < rows + 1; ++i){

  }
}
