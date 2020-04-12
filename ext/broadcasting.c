/*
 * get_index_for_broadcast_element takes prev_shape,
 * and state_array and returns the index
 * for location in flat list of matrix before broadcasting
 *
 */
size_t get_index_for_broadcast_element(size_t* prev_shape, size_t prev_ndims, size_t* state_array, size_t new_dims) {
  size_t* indices = ALLOC_N(size_t, prev_ndims);
  for(size_t i = (new_dims - prev_ndims), index = 0; i < new_dims; ++i, ++index) {
    indices[index] = min(state_array[i], prev_shape[index] - 1);
  }

  size_t new_index = 0;
  size_t* stride = ALLOC_N(size_t, prev_ndims);
  
  size_t val = 1;
  for(size_t i = prev_ndims; i > 0; --i) {
    stride[i - 1] = val;
    val *= prev_shape[i - 1];
  }

  for(size_t i = 0; i < prev_ndims; ++i) {
    new_index += (indices[i] * stride[i]);
  }
  return new_index;
}

/*
 * broadcast_matrix takes the matrix nmat
 * and broadcasts it to new_shape if it's 
 * valid according to broadcasting rules,
 * else raises an error
 */
void broadcast_matrix(nmatrix* nmat, size_t* new_shape, size_t new_ndims) {
  size_t prev_ndims = nmat->ndims;
  size_t* prev_shape = nmat->shape;

  nmat->ndims = new_ndims;
  nmat->shape = ALLOC_N(size_t, new_ndims);
  for(size_t i = 0; i < new_ndims; ++i) {
    nmat->shape[i] = new_shape[i];
  }

  size_t new_count = 1;
  for(size_t i = 0; i < new_ndims; ++i) {
    new_count *= new_shape[i];
  }
  nmat->count = new_count;

  size_t* state_array = ALLOC_N(size_t, new_ndims);
  for(size_t i = 0; i < new_ndims; ++i) {
    state_array[i] = 0;
  }

  switch(nmat->dtype) {
    case nm_bool:
    {
      bool* nmat_elements = (bool*)nmat->elements;

      bool* new_elements = ALLOC_N(bool, new_count);

      for(size_t i = 0; i < new_count; ++i){
        size_t nmat_index = get_index_for_broadcast_element(prev_shape, prev_ndims, state_array, new_ndims);
        new_elements[i] = nmat_elements[nmat_index];

        size_t state_index = (nmat->ndims) - 1;
        while(true){
          size_t curr_index_value = state_array[state_index];

          if(curr_index_value == new_shape[state_index] - 1){
            curr_index_value = 0;
            state_array[state_index] = curr_index_value;
          }
          else{
            curr_index_value++;
            state_array[state_index] = curr_index_value;
            break;
          }

          if(state_index == 0)
            break;

          state_index--;        
        }
      }

      nmat->elements = new_elements;
      break;
    }
    case nm_int:
    {
      int* nmat_elements = (int*)nmat->elements;

      int* new_elements = ALLOC_N(int, new_count);

      for(size_t i = 0; i < new_count; ++i){
        size_t nmat_index = get_index_for_broadcast_element(prev_shape, prev_ndims, state_array, new_ndims);
        new_elements[i] = nmat_elements[nmat_index];

        size_t state_index = (nmat->ndims) - 1;
        while(true){
          size_t curr_index_value = state_array[state_index];

          if(curr_index_value == new_shape[state_index] - 1){
            curr_index_value = 0;
            state_array[state_index] = curr_index_value;
          }
          else{
            curr_index_value++;
            state_array[state_index] = curr_index_value;
            break;
          }

          if(state_index == 0)
            break;

          state_index--;        
        }
      }

      nmat->elements = new_elements;
      break;
    }
    case nm_float32:
    {
      float* nmat_elements = (float*)nmat->elements;

      float* new_elements = ALLOC_N(float, new_count);

      for(size_t i = 0; i < new_count; ++i){
        size_t nmat_index = get_index_for_broadcast_element(prev_shape, prev_ndims, state_array, new_ndims);
        new_elements[i] = nmat_elements[nmat_index];

        size_t state_index = (nmat->ndims) - 1;
        while(true){
          size_t curr_index_value = state_array[state_index];

          if(curr_index_value == new_shape[state_index] - 1){
            curr_index_value = 0;
            state_array[state_index] = curr_index_value;
          }
          else{
            curr_index_value++;
            state_array[state_index] = curr_index_value;
            break;
          }

          if(state_index == 0)
            break;

          state_index--;        
        }
      }

      nmat->elements = new_elements;
      break;
    }
    case nm_float64:
    {
      double* nmat_elements = (double*)nmat->elements;

      double* new_elements = ALLOC_N(double, new_count);

      for(size_t i = 0; i < new_count; ++i){
        size_t nmat_index = get_index_for_broadcast_element(prev_shape, prev_ndims, state_array, new_ndims);
        new_elements[i] = nmat_elements[nmat_index];

        size_t state_index = (nmat->ndims) - 1;
        while(true){
          size_t curr_index_value = state_array[state_index];

          if(curr_index_value == new_shape[state_index] - 1){
            curr_index_value = 0;
            state_array[state_index] = curr_index_value;
          }
          else{
            curr_index_value++;
            state_array[state_index] = curr_index_value;
            break;
          }

          if(state_index == 0)
            break;

          state_index--;        
        }
      }

      nmat->elements = new_elements;
      break;
    }
    case nm_complex32:
    {
      float complex* nmat_elements = (float complex*)nmat->elements;

      float complex* new_elements = ALLOC_N(float complex, new_count);

      for(size_t i = 0; i < new_count; ++i){
        size_t nmat_index = get_index_for_broadcast_element(prev_shape, prev_ndims, state_array, new_ndims);
        new_elements[i] = nmat_elements[nmat_index];

        size_t state_index = (nmat->ndims) - 1;
        while(true){
          size_t curr_index_value = state_array[state_index];

          if(curr_index_value == new_shape[state_index] - 1){
            curr_index_value = 0;
            state_array[state_index] = curr_index_value;
          }
          else{
            curr_index_value++;
            state_array[state_index] = curr_index_value;
            break;
          }

          if(state_index == 0)
            break;

          state_index--;        
        }
      }

      nmat->elements = new_elements;
      break;
    }
    case nm_complex64:
    {
      double complex* nmat_elements = (double complex*)nmat->elements;

      double complex* new_elements = ALLOC_N(double complex, new_count);

      for(size_t i = 0; i < new_count; ++i){
        size_t nmat_index = get_index_for_broadcast_element(prev_shape, prev_ndims, state_array, new_ndims);
        new_elements[i] = nmat_elements[nmat_index];

        size_t state_index = (nmat->ndims) - 1;
        while(true){
          size_t curr_index_value = state_array[state_index];

          if(curr_index_value == new_shape[state_index] - 1){
            curr_index_value = 0;
            state_array[state_index] = curr_index_value;
          }
          else{
            curr_index_value++;
            state_array[state_index] = curr_index_value;
            break;
          }

          if(state_index == 0)
            break;

          state_index--;        
        }
      }

      nmat->elements = new_elements;
      break;
    }
  }
}

/*
 * get_broadcast_shape takes two matrices
 * nmat1, nmat2 and calculates broadcast_shape, broadcast_dims
 * which denotes the shape these 2 matrices should be
 * broadcasted to be compatible for elementwise operations
 */
void get_broadcast_shape(nmatrix* nmat1, nmatrix* nmat2, size_t* broadcast_shape) {
  size_t* shape1 = nmat1->shape;
  size_t* shape2 = nmat2->shape;

  size_t ndims1 = nmat1->ndims;
  size_t ndims2 = nmat2->ndims;

  size_t broadcast_dims = max(ndims1, ndims2);

  if(ndims1 > ndims2) {
    for(size_t i = 0; i < ndims1; ++i) {
      broadcast_shape[i] = shape1[i];
    }
    for(size_t i = 0; i < ndims2; ++i) {
      size_t res_index = (ndims1 - ndims2) + i;
      if(shape1[res_index] != shape2[i] && min(shape1[res_index], shape2[i]) > 1) {
        //raise broadcast compatibility error
      }
      broadcast_shape[res_index] = max(shape1[res_index], shape2[i]);
    }
  }
  else {
    for(size_t i = 0; i < ndims2; ++i) {
      broadcast_shape[i] = shape2[i];
    }
    for(size_t i = 0; i < ndims1; ++i) {
      size_t res_index = (ndims2 - ndims1) + i;
      if(shape1[i] != shape2[res_index] && min(shape1[i], shape2[res_index]) > 1) {
        //raise broadcast compatibility error
      }
      broadcast_shape[res_index] = max(shape1[i], shape2[res_index]);
    }
  }
}

/*
 * broadcast_matrices takes two matrices nmat1, nmat2
 * and broadcasts them against each other.
 * Raises error if matrices are incompatible
 * for broadcasting
 */
void broadcast_matrices(nmatrix* nmat1, nmatrix* nmat2) {
  size_t ndims1 = nmat1->ndims;
  size_t ndims2 = nmat2->ndims;

  size_t broadcast_dims = max(ndims1, ndims2);
  size_t* broadcast_shape = ALLOC_N(size_t, broadcast_dims);

  //check for broadcasting compatibilty
  //and raise error if incompatible

  get_broadcast_shape(nmat1, nmat2, broadcast_shape);

  broadcast_matrix(nmat1, broadcast_shape, broadcast_dims);
  broadcast_matrix(nmat2, broadcast_shape, broadcast_dims);
}

/*
 * Returns a broadcasted matrix created from
 * input matrix and having given shape
 *
 *
 */
VALUE nm_broadcast_to(int argc, VALUE* argv) {
  nmatrix* nmat;
  Data_Get_Struct(argv[0], nmatrix, nmat);

  size_t new_ndims = (size_t)RARRAY_LEN(argv[1]);

  size_t* new_shape = ALLOC_N(size_t, new_ndims);
  for (size_t index = 0; index < new_ndims; index++) {
    new_shape[index] = (size_t)FIX2LONG(RARRAY_AREF(argv[1], index));
  }

  broadcast_matrix(nmat, new_shape, new_ndims);

  return Data_Wrap_Struct(NMatrix, NULL, nm_free, nmat);
}

/*
 * Takes any number of matrices and broadcasts them 
 * against each other and store resulting broadcasted 
 * matrices as array of NMatrix objects
 *
 */
VALUE nm_broadcast_arrays(int argc, VALUE* argv) {
  return Qnil;
}
