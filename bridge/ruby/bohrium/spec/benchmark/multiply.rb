require "bohrium"
require "benchmark"

n = 100_000_000
puts "MULTIPLY"
puts "n = #{n}"

Benchmark.bm(20) do |measure|
  measure.report("zip") do
    ary1 = [2] * n
    ary2 = [3] * n

    ary1.zip(ary2).map { |i| i.reduce(:*) }
  end

  measure.report("zip2") do
    ary1 = [2] * n
    ary2 = [3] * n

    ary1.zip(ary2).map { |f, s| f * s }
  end

  measure.report("idx") do
    ary1 = [2] * n
    ary2 = [3] * n
    ary3 = Array.new(n)

    ary1.each_with_index do |elem, idx|
      ary3[idx] = elem * ary2[idx]
    end
  end

  measure.report("idx<<") do
    ary1 = [2] * n
    ary2 = [3] * n
    ary3 = []

    ary1.each_with_index do |elem, idx|
      ary3 << elem * ary2[idx]
    end
  end

  measure.report("map!") do
    ary1 = [2] * n
    ary2 = [3] * n

    ary1.map!.with_index do |elem, idx|
      elem * ary2[idx]
    end
    ary1
  end

  measure.report("matrix") do
    require "matrix"
    class Matrix
      def element_wise(operator, other)
        Matrix.build(row_size, column_size) do |row, col|
          self[row, col].send(operator, other[row, col])
        end
      end
    end

    m1 = Matrix[[2] * n]
    m2 = Matrix[[3] * n]
    m1.element_wise(:*, m2)
  end

  measure.report("bohrium") do
    ary1 = [2] * n
    ary2 = [3] * n

    a = BhArray.new(ary1)
    b = BhArray.new(ary2)

    a.multiply(b).to_ary # flushes
  end

  measure.report("bohrium ones mult") do
    a = BhArray.ones(n, 1, 2)
    b = BhArray.ones(n, 1, 3)
    a.multiply(b).to_ary # flushes
  end

  measure.report("bohrium ones mult!") do
    a = BhArray.ones(n, 1, 2)
    b = BhArray.ones(n, 1, 3)
    a.multiply!(b)
    a.to_ary # flushes
  end

  measure.report("bohrium ones *") do
    (BhArray.ones(n, 1, 2) * BhArray.ones(n, 1, 3)).to_ary
  end
end
