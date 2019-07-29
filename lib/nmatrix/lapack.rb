module NumRuby::Linalg
  def self.inv(obj)
    if obj.is_a?(NMatrix)
      return obj.invert
    end
  end

  def self.dot(lha, rha)
    lha.dot(rha)
  end

  def self.norm

  end

  def self.solve

  end

  def self.det

  end

  def self.least_square

  end

  def self.pinv

  end

  def self.kronecker_prod

  end

  def self.eig

  end

  def self.eigh

  end

  def self.eigvalsh

  end

  def self.lu

  end

  def self.lu_factor

  end

  def self.lu_solve

  end

  def self.svd

  end

  def self.svdvals

  end

  def self.diagsvd

  end

  def self.orth

  end

  def self.cholesky

  end

  def self.cholesky_solve

  end

  def self.qr

  end
end