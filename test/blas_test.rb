require_relative 'test_helper'

class NMatrix::BlasTest < Minitest::Test

  def setup
    @left = NMatrix.new  [2,2],[2.2, 2.2, 4, 5]
    @right = NMatrix.new [2,2],[2, 2, 2, 2]
    @dtypes = [:nm_float32, :nm_float64]
  end

  def test_dot
    @dtypes.each do |dtype|
      left = NMatrix.new  [2,2],[2.2, 2.2, 4, 5], dtype
      right = NMatrix.new [2,2],[2, 2, 2, 2], dtype
      result = NMatrix.new  [2,2], [8.8, 8.8, 18.0, 18.0], dtype
      answer = left.dot right
      assert_equal answer, result
    end
  end

  def test_norm
    @dtypes.each do |dtype|
      vector = NMatrix.new  [2,1], [3.3, 4.8], dtype
      assert_in_delta vector.norm, 5.8249, 1e-3
    end
  end

end
