require 'test_helper'

class NMatrix::LapackTest < Minitest::Test

  def setup
    @left = NMatrix.new  [2,2],[2.2, 2.2, 4, 5]
    @right = NMatrix.new [2,2],[2, 2, 2, 2]
  end

  def test_invert

  end

  def test_solve

  end

  def test_det

  end

  def test_least_square

  end

  def test_pinv

  end

  def test_kronecker_prod

  end

  def test_eig

  end

  def test_eigh

  end

  def test_eigvalsh

  end

  def test_lu

  end

  def test_lu_factor

  end

  def test_lu_solve

  end

  def test_svd

  end

  def test_svdvals

  end

  def test_diagsvd

  end

  def test_orth

  end

  def test_cholesky

  end

  def test_cholesky_solve

  end

  def test_qr

  end

end
