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

  # Matrix Decomposition


  def self.lu(matrix)

  end

  def self.lu_factor(matrix)

  end

  def self.lu_solve(matrix, rhs_val)

  end

  # Computes the QR decomposition of matrix.
  # Args:
  # - input matrix, type: NMatrix
  # - mode, type: String
  # - pivoting, type: Boolean
  def self.svd(matrix)

  end

  def self.svdvals(matrix)

  end

  def self.diagsvd(matrix)

  end

  def self.orth(matrix)

  end

  def self.cholesky(matrix)

  end

  def self.cholesky_solve(matrix)

  end

  # Computes the QR decomposition of matrix.
  # Args:
  # - input matrix, type: NMatrix
  # - mode, type: String
  # - pivoting, type: Boolean
  def self.qr(matrix, mode: "full", pivoting: false)
    if not ['full', 'r', 'economic', 'raw'].include?(mode.downcase)
      raise("Invalid mode. Should be one of ['full', 'r', 'economic', 'raw']")
    end
    if not matrix.is_a?(NMatrix)
      raise("Invalid matrix. Not of type NMatrix")
    end
    if matrix.dim != 2
      raise("Invalid shape of matrix. Should be 2.")
    end
    m, n = matrix.shape

    if pivoting == false
      qr, tau = NumRuby::Lapack.geqrf(matrix)
      return [qr, tau]
    end
  end
end