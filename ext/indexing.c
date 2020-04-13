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