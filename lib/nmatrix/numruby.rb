module NumRuby

  def self.array(shape, elements, options = {})

    if(options[:dtype])
      dtype = options[:dtype]
    else
      dtype = :nm_float64
    end

    NMatrix.new(shape, elements, dtype)
  end

  def self.arange(len)
    elements = Array(0..len)
    NMatrix.new([len,1], elements, :nm_int)
  end

  # unary operations
  def self.sin(obj)
    if obj.is_a?(NMatrix)
      obj.sin()
    else
      Math.sin(obj)
    end
  end

  def self.append(a, b)
    self.vstack([a,b])
  end

  def self.vstack(objs)
    return nil unless objs.is_a?(Array)
    rows = objs[0].shape[0] * objs.length
    cols = objs[0].shape[1]
    result = NumRuby.zeros([rows, cols])
    objs.each do |obj|
      #check num_cols mismatch
      # (0..objs.shape[0]).each
    end

    # (0..result.shape[0]).each do |row_index|
    #   result[row_index, *] = objs[x][row_index, *]
    #   index = (row_index == rows ? 0 : index + 1)
    # end
    result
  end

  def self.hstack(objs)
    return nil unless objs.is_a?(Array)
    rows = objs[0].shape[0]
    cols = objs[0].shape[1] * objs.length
    result = NumRuby.zeros([rows, cols])
    objs.each do |obj|
      #check num_rows mismatch
    end

    # (0..result.shape[1]).each do |col_index|
    #   result[col_index, *] = objs[x][col_index, *]
    #   index = (col_index == cols ? 0 : index + 1)
    # end
    result
  end

  def self.tri(N, M=None, k=0, nm_dtype=float)
    if M is None:
        M = N

      m = greater_equal.outer(arange(N, nm_dtype=nm_int(0, N)),
                              arange(-k, M-k, nm_dtype=nm_int(-k, M - k)))

      # Avoid making a copy if the requested type is already bool
      m = m.stype(nm_dtype, copy=False)
      return m
    end
  end

  def tril(m, k=0)
    m = Array(m)
    mask = tri(*m.shape[-2:], k=k, nm_dtype=nm_bool)
  end

  def triu(m, k=0)
    m = Array(m)
    mask = tri(*m.shape[-2:], k=k-1, nm_dtype=nm_bool)
  end

end
