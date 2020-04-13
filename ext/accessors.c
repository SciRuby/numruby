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