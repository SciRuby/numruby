class NMatrix

  # Printing the NMatrix object
  #
  # @param q
  # @return [String] inspect
  #
  # @example
  #  TODO
  def pretty_print(q)
    if self.dim == 1
      q.group(0, "\n[", "]") do
        q.seplist(self.elements, -> { q.text ", " }, :each) do |v|
          q.text v.inspect
        end
      end
    elsif self.dim == 2
      # iterate through the whole matrix and find the longest number for each column
      longest = Array.new(self.shape[1], 0)

      for col_index in 0...self.shape[1] do
        j = col_index
        self.shape[0].times do
          elem_len           = self.elements[j].inspect.size
          longest[col_index] = elem_len if longest[col_index] < elem_len
          j += self.shape[1]
        end
      end

      q.group(0, "\n[\n", "]") do
        for row_index in 0...self.shape[0] do
          i = (row_index * self.shape[1])
          current_row = Array.new
          self.shape[1].times do
            current_row.push(self.elements[i])
            i += 1
          end
          q.group(1, " [", "]") do
            q.seplist(current_row, -> { q.text ", " }, :each_with_index) do |v,j|
              q.text v.inspect.rjust(longest[j])
            end
          end
          q.comma_breakable unless row_index + 1 == self.shape[0]
          q.text "\n"
        end
      end
    elsif self.dim == 3
      q.group(0, "\n[\n", "]") do
        for layer_index in 0...self.shape[2] do
          # iterate through the whole matrix and find the longest number for each column
          longest = Array.new(self.shape[1], 0)

          for col_index in 0...self.shape[1] do
            j = (col_index * self.shape[2]) + layer_index
            self.shape[0].times do
              elem_len           = self.elements[j].inspect.size
              longest[col_index] = elem_len if longest[col_index] < elem_len
              j += (self.shape[1] * self.shape[2])
            end
          end

          q.group(1, " [\n", " ]") do
            for row_index in 0...self.shape[0] do
              i = (row_index * self.shape[1] * self.shape[2]) + layer_index
              current_row = Array.new
              self.shape[1].times do
                current_row.push(self.elements[i])
                i += self.shape[2]
              end
              q.group(2, "  [", "]") do
                q.seplist(current_row, -> { q.text ", " }, :each_with_index) do |v,j|
                  q.text v.inspect.rjust(longest[j])
                end
              end
              q.text "," unless row_index + 1 == self.shape[0]
              q.text "\n"
            end
          end
          q.text "," unless layer_index + 1 == self.shape[2]
          q.text "\n"
        end
      end
    else
      self.inspect.pretty_print(q)
    end
  end

  def to_a
    return self.elements
  end

  def _dump data
    [
      self.dim,
      self.dtype,
      self.stype,
      self.shape,
      self.elements,
    ].join(":")
  end

  def self._load serial
    data = serial.split(":")
    dim = data[0].to_i
    dtype = data[1].to_sym
    stype = data[2].to_sym
    shape = data[3..(3+dim-1)]
    elements = data[(3+dim)..data.length-1]
    parsed_elements = []
    
    if dtype == :nm_bool 
      elements.each do |e|
        parsed_elements.push(e == "true" ? true : false)
      end
    elsif dtype == :nm_float64
      elements.each do |e|
        parsed_elements.push(e.to_f)
      end
    else 
      # Convert to Integer
      elements.each do |e|
        parsed_elements.push(e.to_i)
      end
    end
    parsed_shape = []
    shape.each do |e| 
      parsed_shape.push(e.to_i)
    end
    self.new(parsed_shape, parsed_elements, dtype)
  end

  def inspect #:nodoc:
    original_inspect = super()
    original_inspect = original_inspect[0...original_inspect.size-1]
    original_inspect + " " + inspect_helper.join(" ") + ">"
  end

  protected

  def inspect_helper #:nodoc:
    ary = []
    ary << "shape:[#{shape.join(',')}]" << "dtype:#{dtype}" << "stype:#{stype}"
    ary
  end
end
