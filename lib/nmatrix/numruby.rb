module NumRuby

  def self.array(shape, elements, options = {})

    if(options[:dtype])
      dtype = options[:dtype]
    end

    NMatrix.new(shape, elements, dtype)
  end


end
