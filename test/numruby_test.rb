require 'test_helper'

class NumRuby::CreationTest < Minitest::Test

  def setup
    @i = NumRuby.array [2,2],[1, 4.2, 3,4], dtype: :nm_float64
  end

  def test_shape
    assert_equal [2,2], @i.shape
  end

  def test_dtype
    assert_equal :nm_float64, @i.dtype
  end

  def test_elements
    assert_equal [1, 4.2, 3, 4], @i.elements
  end

  def test_arange
    x = NumRuby.arange 5
    assert_equal x.elements, [0,1,2,3,4]
    assert_equal x.shape, [5, 1]
    assert_equal x.dtype, :nm_int
  end

  def test_sin
    x = NumRuby.sin(5)
    assert_equal x, Math.send(:sin, 5)

    x = NumRuby.sin(@i)
    assert_equal x, @i.sin
  end

  def test_inv
    assert_equal NumRuby::Linalg.inv(@i), @i.invert
  end

  def test_append
    x = NumRuby.array [2,2],[0, 0, 0, 0]
    y = NumRuby.array [2,2],[0, 0, 0, 0]
    result = NumRuby.append(x, y)
    assert_equal result, NumRuby.array([4,2], [0, 0, 0, 0, 0, 0, 0, 0])
  end
end
