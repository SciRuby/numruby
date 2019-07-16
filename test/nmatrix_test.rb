require 'test_helper'

class NMatrix::CreationTest < Minitest::Test

  def setup
    @i = NMatrix.new [2,2],[1, 4.2, 3, 4]
    @b = NMatrix.new [2,2],[true, true, false, true], :nm_bool
    @m = NMatrix.new [2,2,2],[1, 2, 3, 4, 5, 6, 7, 8]
    @n = NMatrix.new [2,1,2],[1, 2, 3, 4]
    @s = NMatrix.new [2, 2],[1, 2, 3, 4]
    @s_int = NMatrix.new [2, 2],[1, 2, 3, 4], :nm_int
  end

  def test_dims
    assert_equal [2,2], @i.shape
    assert_equal [2,2], @b.shape
    assert_equal [2,2,2], @m.shape
    assert_equal [2,1,2], @n.shape
  end

  def test_dtype
    assert_equal :nm_float64, @i.dtype
  end

  def test_stype
    assert_equal :nm_dense, @i.stype
  end

  def test_elements
    assert_equal [1, 4.2, 3, 4], @i.elements
    assert_equal [true, true, false, true], @b.elements
    assert_equal [1, 2, 3, 4, 5, 6, 7, 8], @m.elements
    assert_equal [1, 2, 3, 4], @n.elements
  end

  def test_dim
    assert_equal 2, @i.dim
    assert_equal 3, @m.dim
  end

  def test_accessor_get
    assert_equal @i[0,1], 4.2
    assert_equal @m[0,0,1], 2
    assert_equal @n[1,0,0], 3
    assert_equal @n[1,0,1], 4
  end

  def test_accessor_set
    @i[0,0] = 6
    assert_equal @i[0,0], 6
    @m[1,1,0] = 11
    assert_equal @m[1,1,0], 11
    @n[0,0,1] = 12
    assert_equal @n[0,0,1], 12
  end

  def test_slicing
    assert_equal @m[0, 0..1, 0..1], @s
    assert_equal @m[0, 0..1, 0..1], @s_int
  end

end
