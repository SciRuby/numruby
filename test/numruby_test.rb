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

end
