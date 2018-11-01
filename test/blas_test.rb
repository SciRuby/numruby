require 'test_helper'

class NMatrix::BlasTest < Minitest::Test

  def setup
    @left = NMatrix.new  [2,2],[2.2, 2.2, 4, 5]
    @right = NMatrix.new [2,2],[2, 2, 2, 2]
  end

  def test_dot
    result = NMatrix.new  [2,2], [8.8, 8.8, 18.0, 18.0]
    answer = @left.dot @right
    assert_equal answer.elements, result.elements
  end

end
