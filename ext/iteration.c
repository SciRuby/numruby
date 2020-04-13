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