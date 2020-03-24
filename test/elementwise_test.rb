require_relative 'test_helper'

class NMatrix::ElementWiseTest < Minitest::Test

  def setup
    @left = NMatrix.new  [2,2],[  1, 4.2,  3, 4.2]
    @right = NMatrix.new [2,2],[1.9, 5.2, 3.1,  4]

    @complex_left = NMatrix.new  [2,2],[1 + 2i, 4.2 + 5i, 3 + 0i, 4.2 - 7i], :nm_complex32
    @complex_right = NMatrix.new  [2,2],[1 + 2i, 4.2 + 5i, 3 + 0i, 4.2 - 7i], :nm_complex32

    @boolean_left = NMatrix.new  [2,2],[true, false, true, false], :nm_bool
    @boolean_right = NMatrix.new [2,2],[true, true, false, false], :nm_bool
  end

  def test_add
    result = NMatrix.new  [2,2],[2.9, 9.4, 6.1, 8.2]
    answer = @left + @right
    assert_equal answer, result

    result = NMatrix.new  [2,2],[1.1, 4.3, 3.1, 4.3]
    answer = @left + 0.1
    assert_equal answer, result
  end

  def test_add_bool
    result = NMatrix.new  [2,2],[true, true, true, false], :nm_bool
    answer = @boolean_left + @boolean_right
    assert_equal answer, result

    result = NMatrix.new  [2,2],[true, true, true, true], :nm_bool
    answer = @boolean_left + true
    assert_equal answer, result

    result = NMatrix.new  [2,2],[true, false, true, false], :nm_bool
    answer = @boolean_left + false
    assert_equal answer, result
  end

  def test_add_complex
    result = NMatrix.new  [2,2],[2.9, 9.4, 6.1,   8.2]
    answer = @complex_left + @complex_right
    skip("equality assertion missing for complex dtype")
  end

  def test_subtract
    result = NMatrix.new  [2,2],[-0.9, -1, -0.1,   0.2]
    answer = @left - @right
    assert_equal answer, result
  end

  def test_sin
    result = NMatrix.new [2,2], @left.elements.map{ |x| Math.send(:sin, x) }
    answer = @left.sin
    assert_equal answer, result
  end

  def test_cos
    result = NMatrix.new [2,2], @left.elements.map{ |x| Math.send(:cos, x) }
    answer = @left.cos
    assert_equal answer, result
  end

  def test_tan
    result = NMatrix.new [2,2], @left.elements.map{ |x| Math.send(:tan, x) }
    answer = @left.tan
    assert_equal answer, result
  end

end
