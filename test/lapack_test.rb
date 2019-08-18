require_relative 'test_helper'

class NMatrix::LapackTest < Minitest::Test

  def setup
    @input = NMatrix.new [2,2], [2, -1, -4, 3]
    @matrix1 = NMatrix.new [2, 3], [4, -1, 4, 0, 7, -5]
    @matrix2 = NMatrix.new [3, 2], [4, -1, 4, 0, 7, -5]
    @matrix3 = NMatrix.new [3, 3], [1, 0, -5, 3, 5, 2, 0, -1, -1]
    @left = NMatrix.new  [2,2],[2.2, 2.2, 4, 5]
    @right = NMatrix.new [2,2],[2, 2, 2, 2]
    @dtypes = [:nm_float64]
    @dtypes2 = [:nm_float64, :nm_float32]
  end

  def test_invert
    @dtypes.each do |dtype|
      input = NMatrix.new [2,2], [2, -1, -4, 3], dtype
      result = NMatrix.new [2, 2], [1.5, 0.5, 2.0, 1.0], dtype
      assert_equal input.invert, result
    end
  end

  def test_solve
    @dtypes.each do |dtype|
      input = NMatrix.new [2,2], [2, -1, -4, 3], dtype
      rhs = NMatrix.new [2, 1], [4, 7], dtype
      solution = NMatrix.new [2,1], [9.5, 15], dtype
      assert_equal input.solve(rhs), solution
    end
  end

  def test_det
    @dtypes2.each do |dtype|
      input = NMatrix.new [2,2], [2, -1, -4, 3], dtype
      assert_equal input.det, 2
    end
  end

  def test_least_square
    @dtypes2.each do |dtype|
      input = NMatrix.new [2,2], [2, -1, -4, 3], dtype
      rhs = NMatrix.new [2, 1], [4, 7], dtype
      solution = NMatrix.new [2,1], [9.5, 15], dtype
      assert_equal input.least_square(rhs), solution
    end
  end

  def test_geqrf
    qr, tau = NumRuby::Lapack.geqrf(@matrix1)
    qr_soln = NMatrix.new [2, 3], [4.0, -1.0, 4.0, 0.0, 7.0, -5.0]
    tau_soln = NMatrix.new [2], [0.0, 0.0]
    assert_equal qr, qr_soln
    assert_equal tau, tau_soln

    qr, tau = NumRuby::Lapack.geqrf(@matrix2)
    qr_soln = NMatrix.new [3, 2], [-9.0, 4.333333, 0.307692, -2.687419, 0.538461, -0.491678]
    tau_soln = NMatrix.new [2], [1.444444, 1.610632]
    assert_equal qr, qr_soln
    assert_equal tau, tau_soln
    
    qr, tau = NumRuby::Lapack.geqrf(@matrix3)
    qr_soln = NMatrix.new [3, 3], [-3.162277, -4.743416, -0.316227, 0.720759, -1.870828, -5.077963, 0.0, -0.289689, 2.028370]
    tau_soln = NMatrix.new [3], [1.316227, 1.845154, 0.0]
    assert_equal qr, qr_soln
    assert_equal tau, tau_soln
  end

  def test_orgqr
    qr = NMatrix.new [3, 2], [-9.0, 4.333333, 0.307692, -2.687419, 0.538461, -0.491678]
    tau = NMatrix.new [2], [1.444444, 1.610632]
    q = NumRuby::Lapack.orgqr(qr, tau)
    q_soln = NMatrix.new [3, 2], [-0.444444, -0.344540, -0.444444, -0.716645, -0.777777, 0.606392]
    assert_equal q, q_soln
  end

  def test_geqp3
    ## doesn't give same output on each run
  end

  def test_potrf
    matrix = NMatrix.new [3, 3], [2, -1, 0, -1, 2, -1, 0, -1, 2]

    c = NumRuby::Lapack.potrf(matrix, true)
    c_soln = NMatrix.new [3, 3], [1.414213, -1.0, 0.0, -0.707106, 1.224744, -1.0, 0.0, -0.816496, 1.154700]
    assert_equal c, c_soln

    c = NumRuby::Lapack.potrf(matrix, false)
    c_soln = NMatrix.new [3, 3], [1.414213, -0.707106, 0.0, -1.0, 1.224744, -0.816496, 0.0, -1.0, 1.154700]
    assert_equal c, c_soln
  end

  def test_potrs
    matrix = NMatrix.new [3, 3], [2, -1, 0, -1, 2, -1, 0, -1, 2]
    b = NMatrix.new [3], [1, 2, 3]

    c = NumRuby::Lapack.potrf(matrix, true)
    x = NumRuby::Lapack.potrs(c, b, true)
    x_soln = NMatrix.new [3], [2.5, 4.0, 3.5]
    assert_equal x, x_soln

    c = NumRuby::Lapack.potrf(matrix, false)
    x = NumRuby::Lapack.potrs(c, b, false)
    x_soln = NMatrix.new [3], [2.5, 4.0, 3.5]
    assert_equal x, x_soln
  end

  def test_gesdd
    # TODO: complete nm_gesdd
  end

  def test_getrf
    lu, ipiv = NumRuby::Lapack.getrf(@matrix3)
    lu_soln = NMatrix.new [3, 3], [3.0, 5.0, 2.0, 0.333333, -1.666666, -5.666666, 0.0, 0.6, 2.4]
    assert_equal lu, lu_soln

    a = NMatrix.new [2, 2], [1, 2, 3, 4]
    lu, ipiv = NumRuby::Lapack.getrf(a)
    lu_soln = NMatrix.new [2, 2], [3.0, 4.0, 0.333333, 0.666666]
    assert_equal lu, lu_soln
  end

  def test_getrs
    matrix = NMatrix.new [3, 3], [2, -1, 0, -1, 2, -1, 0, -1, 2]
    b = NMatrix.new [3], [1, 2, 3]

    lu, ipiv = NumRuby::Lapack.getrf(matrix)
    x = NumRuby::Lapack.getrs(lu, ipiv, b, 0)
    x_soln = NMatrix.new [3], [2.5, 4.0, 3.5]
    assert_equal x, x_soln
  end

  def test_getri
    matrix = NMatrix.new [3, 3], [2, -1, 0, -1, 2, -1, 0, -1, 2]

    lu, ipiv = NumRuby::Lapack.getrf(matrix)
    inv = NumRuby::Lapack.getri(lu, ipiv)
    inv_soln = NMatrix.new [3, 3], [0.75, 0.5, 0.25, 0.5, 1.0, 0.5, 0.25, 0.5, 0.75]
    assert_equal inv, inv_soln
  end

  def test_gelss
    # TODO: implement nm_gelss
  end

  def test_posv
    
  end

  def test_gesv

  end

  def test_lange

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
