module NumRuby::Linalg
  def self.inv(matrix)
    if not matrix.is_a?(NMatrix)
      raise("Invalid matrix. Not of type NMatrix.")
    end
    if matrix.dim != 2
      raise("Invalid shape of matrix. Should be 2.")
    end
    if matrix.shape[0] != matrix.shape[1]
      raise("Invalid shape. Expected square matrix.")
    end
    m, n = matrix.shape

    lu, ipiv = NumRuby::Lapack.getrf(matrix)
    inv_a = NumRuby::Lapack.getri(lu, ipiv)

    return inv_a
  end

  def self.dot(lha, rha)
    lha.dot(rha)
  end

  def self.norm

  end

  def self.solve(a, b, sym_pos: False, lower: False, assume_a: "gen", transposed: False)
    # TODO: implement this and remove NMatrix.solve
  end

  def self.det(matrix)
    if not matrix.is_a?(NMatrix)
      raise("Invalid matrix. Not of type NMatrix.")
    end
    if matrix.dim != 2
      raise("Invalid shape of matrix. Should be 2.")
    end
    if matrix.shape[0] != matrix.shape[1]
      raise("Invalid shape. Expected square matrix.")
    end

    return matrix.det
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

  # Computes the SVD decomposition of matrix.
  # Args:
  # - input matrix, type: NMatrix
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

  # Computes QR decomposition of a matrix.
  #
  # Calculates the decomposition A = Q*R where Q is unitary/orthogonal and R is upper triangular.
  #
  # Args:
  # - matrix, type: NMatrix
  #     Matrix to be decomposed
  # - mode, type: String
  #     Determines what information is to be returned: either both Q and R
  #     ('full', default), only R ('r') or both Q and R but computed in
  #     economy-size ('economic', see Notes). The final option 'raw'
  #     (added in Scipy 0.11) makes the function return two matrices
  #     (Q, TAU) in the internal format used by LAPACK.
  # - pivoting, type: Boolean
  #     Whether or not factorization should include pivoting for rank-revealing
  #     qr decomposition. If pivoting, compute the decomposition
  #     A*P = Q*R as above, but where P is chosen such that the diagonal
  #     of R is non-increasing.
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

    if pivoting == true
      qr, tau, jpvt = NumRuby::Lapack.geqp3(matrix)
      jpvt -= 1
    else
      qr, tau = NumRuby::Lapack.geqrf(matrix)
    end

    # calculate R here for both pivot true & false

    if ['economic', 'raw'].include?(mode.downcase) or m < n
      r = NumRuby.triu(matrix)
    else
      r = NumRuby.triu(matrix[0...n, 0...n])
    end

    if pivoting == true
      rj = r, jpvt
    else
      rj = r
    end

    if mode == 'r'
      return rj
    elsif mode == 'raw'
      return [qr, tau]
    end

    if m < n
      q = NumRuby::Lapack.orgqr(qr[0...m, 0...m], tau)
    elsif mode == 'economic'
      q = NumRuby::Lapack.orgqr(qr, tau)
    else
      # TODO: Implement slice view and set slice
      q = NumRuby::Lapack.orgqr(qr, tau)
    end

    return q, rj
  end
end