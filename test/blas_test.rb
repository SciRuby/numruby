require 'test_helper'

class NMatrix::BlasTest < Minitest::Test

  def setup
    @left = NMatrix.new  [2,2],[2.2, 2.2, 4, 5]
    @right = NMatrix.new [2,2],[2, 2, 2, 2]
  end

  def test_dot
    result = NMatrix.new  [2,2], [8.8, 8.8, 18.0, 18.0]
    answer = @left.dot @right
    assert_equal answer, result
  end

  def test_norm
    vector = NMatrix.new  [2,1], [3, 4]
    assert_equal vector.norm, 5
  end

end
