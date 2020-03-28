require_relative 'test_helper'

class NumRuby::StatisticsTest < Minitest::Test

  def setup
    @matrix = NMatrix.new  [2,2],[2.2, 2.2, 4, 5]
    @weight = NMatrix.new [2,1],[1, 2]
  end

  def test_average
    result = NMatrix.new  [2,1], [3.3, 7.0]
    assert_equal NumRuby.average(@matrix, 0, @weight), result
    result = NMatrix.new  [2,1], [5.1, 6.1]
    assert_equal NumRuby.average(@matrix, 1, @weight), result
  end

  def test_norm
    vector = NMatrix.new  [2,1], [3, 4]
    assert_equal vector.norm, 5
  end

end
