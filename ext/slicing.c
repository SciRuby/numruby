/*
 * converts Range objects to corresponding
 * lower limit and upper limit and put them in size_t variables
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
void get_slice(nmatrix* nmat, size_t* lower, size_t* upper, nmatrix_buffer* slice){
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

  VALUE* state_array = ALLOC_N(VALUE, nmat->ndims);
  for(size_t i = 0; i < nmat->ndims; ++i){
    state_array[i] = SIZET2NUM(lower[i]);
  }

  // for float64
  double* nmat_elements = (double*)nmat->elements;
  size_t start_index = get_index(nmat, state_array);  // slice first element index in elements array
  slice->buffer_ele_start_ptr = (nmat_elements + start_index);

  //mark elements that are inside the slice
  //and copy them to elements array of slice

  // below code is moved to get_elements func
  // as on using nmatrix buffer,
  // the elements doesn't need to be copied
  // but the iteration of elements needs to
  // be done using the starting element of buffer
  // and original shape strides


  // switch (nmat->dtype){
  //   case nm_bool:
  //   {
  //     bool* nmat_elements = (bool*)nmat->elements;

  //     bool* slice_elements = ALLOC_N(bool, slice->count);

  //     for(size_t i = 0; i < slice->count; ++i){
  //       size_t nmat_index = get_index(nmat, state_array);
  //       slice_elements[i] = nmat_elements[nmat_index];

  //       size_t state_index = (nmat->ndims) - 1;
  //       while(true){
  //         size_t curr_index_value = NUM2SIZET(state_array[state_index]);

  //         if(curr_index_value == upper[state_index]){
  //           curr_index_value = lower[state_index];
  //           state_array[state_index] = SIZET2NUM(curr_index_value);
  //         }
  //         else{
  //           curr_index_value++;
  //           state_array[state_index] = SIZET2NUM(curr_index_value);
  //           break;
  //         }  

  //         state_index--;        
  //       }
  //     }

  //     slice->elements = slice_elements;
  //     break;
  //   }
  //   case nm_int:
  //   {
  //     int* nmat_elements = (int*)nmat->elements;

  //     int* slice_elements = ALLOC_N(int, slice->count);

  //     for(size_t i = 0; i < slice->count; ++i){
  //       size_t nmat_index = get_index(nmat, state_array);
  //       slice_elements[i] = nmat_elements[nmat_index];

  //       size_t state_index = (nmat->ndims) - 1;
  //       while(true){
  //         size_t curr_index_value = NUM2SIZET(state_array[state_index]);

  //         if(curr_index_value == upper[state_index]){
  //           curr_index_value = lower[state_index];
  //           state_array[state_index] = SIZET2NUM(curr_index_value);
  //         }
  //         else{
  //           curr_index_value++;
  //           state_array[state_index] = SIZET2NUM(curr_index_value);
  //           break;
  //         }  

  //         state_index--;        
  //       }
  //     }

  //     slice->elements = slice_elements;
  //     break;
  //   }
  //   case nm_float64:
  //   {
  //     double* nmat_elements = (double*)nmat->elements;

  //     double* slice_elements = ALLOC_N(double, slice->count);

  //     for(size_t i = 0; i < slice->count; ++i){
  //       size_t nmat_index = get_index(nmat, state_array);
  //       slice_elements[i] = nmat_elements[nmat_index];

  //       size_t state_index = (nmat->ndims) - 1;
  //       while(true){
  //         size_t curr_index_value = NUM2SIZET(state_array[state_index]);

  //         if(curr_index_value == upper[state_index]){
  //           curr_index_value = lower[state_index];
  //           state_array[state_index] = SIZET2NUM(curr_index_value);
  //         }
  //         else{
  //           curr_index_value++;
  //           state_array[state_index] = SIZET2NUM(curr_index_value);
  //           break;
  //         }  

  //         state_index--;        
  //       }
  //     }

  //     slice->elements = slice_elements;
  //     break;
  //   }
  //   case nm_float32:
  //   {
  //     float* nmat_elements = (float*)nmat->elements;

  //     float* slice_elements = ALLOC_N(float, slice->count);

  //     for(size_t i = 0; i < slice->count; ++i){
  //       size_t nmat_index = get_index(nmat, state_array);
  //       slice_elements[i] = nmat_elements[nmat_index];

  //       size_t state_index = (nmat->ndims) - 1;
  //       while(true){
  //         size_t curr_index_value = NUM2SIZET(state_array[state_index]);

  //         if(curr_index_value == upper[state_index]){
  //           curr_index_value = lower[state_index];
  //           state_array[state_index] = SIZET2NUM(curr_index_value);
  //         }
  //         else{
  //           curr_index_value++;
  //           state_array[state_index] = SIZET2NUM(curr_index_value);
  //           break;
  //         }  

  //         state_index--;        
  //       }
  //     }

  //     slice->elements = slice_elements;
  //     break;
  //   }
  //   case nm_complex32:
  //   {
  //     float complex* nmat_elements = (float complex*)nmat->elements;

  //     float complex* slice_elements = ALLOC_N(float complex, slice->count);

  //     for(size_t i = 0; i < slice->count; ++i){
  //       size_t nmat_index = get_index(nmat, state_array);
  //       slice_elements[i] = nmat_elements[nmat_index];

  //       size_t state_index = (nmat->ndims) - 1;
  //       while(true){
  //         size_t curr_index_value = NUM2SIZET(state_array[state_index]);

  //         if(curr_index_value == upper[state_index]){
  //           curr_index_value = lower[state_index];
  //           state_array[state_index] = SIZET2NUM(curr_index_value);
  //         }
  //         else{
  //           curr_index_value++;
  //           state_array[state_index] = SIZET2NUM(curr_index_value);
  //           break;
  //         }  

  //         state_index--;        
  //       }
  //     }

  //     slice->elements = slice_elements;
  //     break;
  //   }
  //   case nm_complex64:
  //   {
  //     double complex* nmat_elements = (double complex*)nmat->elements;

  //     double complex* slice_elements = ALLOC_N(double complex, slice->count);

  //     for(size_t i = 0; i < slice->count; ++i){
  //       size_t nmat_index = get_index(nmat, state_array);
  //       slice_elements[i] = nmat_elements[nmat_index];

  //       size_t state_index = (nmat->ndims) - 1;
  //       while(true){
  //         size_t curr_index_value = NUM2SIZET(state_array[state_index]);

  //         if(curr_index_value == upper[state_index]){
  //           curr_index_value = lower[state_index];
  //           state_array[state_index] = SIZET2NUM(curr_index_value);
  //         }
  //         else{
  //           curr_index_value++;
  //           state_array[state_index] = SIZET2NUM(curr_index_value);
  //           break;
  //         }  

  //         state_index--;        
  //       }
  //     }

  //     slice->elements = slice_elements;
  //     break;
  //   }
  // }

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