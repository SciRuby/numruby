$LOAD_PATH.unshift File.expand_path('../../lib', __FILE__)
require 'nmatrix'

require 'benchmark'
require 'json'

class ResultCollect


  def self.generate

    iters = 0
    result = {}

    result[:addition] = []
    result[:subtraction] = []
    result[:mat_mult] = []

    shapeArray = [
                  [10,10],
                  [50,50],
                  [100,100],
                  [500,500],
                  [1000,1000],
                  [2000,2000],
                  [3000,3000],
                  [4000,4000],
                  [5000,5000]
                ]

    shapeArray.each do |shape|
      elements1 = Array.new(shape[0]*shape[1]) { rand(1...999999) }
      elements2 = Array.new(shape[0]*shape[1]) { rand(1...999999) }
      nmatrix1 = NMatrix.new(shape, elements1)
      nmatrix2 = NMatrix.new(shape, elements2)

      iters.times {nmatrix1 + nmatrix2}
      result[:addition] << [ shape[0]*shape[1], Benchmark.measure{nmatrix1 + nmatrix2}.to_s.tr('()', '').split(" ")[3].to_f ]

      iters.times {nmatrix1 - nmatrix2}
      result[:subtraction] << [ shape[0]*shape[1], Benchmark.measure{nmatrix1 - nmatrix2}.to_s.tr('()', '').split(" ")[3].to_f ]

      iters.times {nmatrix1.dot(nmatrix2)}
      result[:mat_mult] << [ shape[0]*shape[1], Benchmark.measure{nmatrix1.dot(nmatrix2)}.to_s.tr('()', '').split(" ")[3].to_f ]
    end
    puts result

  end
end

ResultCollect.generate
