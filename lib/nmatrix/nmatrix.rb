class NMatrix

  def pretty_print(q)
    if self.shape.size > 1 and self.shape[1] > 100
      self.inspect.pretty_print(q)
    end
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
