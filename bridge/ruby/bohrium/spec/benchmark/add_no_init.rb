require "benchmark"

n = 100_000_000
puts "ADD"
puts "n = #{n}"

ARY1 = [2] * n
ARY2 = [3] * n

ARY1B = [2] * n

ARY3 = Array.new(n)
ARY3B = []

require "matrix"
M1 = Matrix[[2] * n]
M2 = Matrix[[3] * n]
M12D = Matrix[[[2] * n]]
M22D = Matrix[[[3] * n]]

require "numo/narray"
M1N1 = Numo::Int64.new(n).fill(2)
M2N1 = Numo::Int64.new(n).fill(3)
M1N2 = Numo::Int64.new(n, 1).fill(2)
M2N2 = Numo::Int64.new(n, 1).fill(3)

require "bohrium"
BH1 = BhArray.ones(n, 1, 2)
BH2 = BhArray.ones(n, 1, 3)

Benchmark.bmbm(20) do |measure|
  measure.report("zip") do
    ARY1.zip(ARY2).map { |i| i.reduce(:+) }[0..10].to_s
  end

  measure.report("zip_lazy") do
    ARY1.lazy.zip(ARY2).map { |i| i.reduce(:+) }.to_a[0..10].to_s
  end

  measure.report("zip_sum") do
    ARY1.zip(ARY2).map(&:sum)[0..10].to_s
  end

  measure.report("zip2") do
    ARY1.zip(ARY2).map { |a, b| a + b }[0..10].to_s
  end

  measure.report("idx") do
    ARY1.each_with_index do |elem, idx|
      ARY3[idx] = elem + ARY2[idx]
    end

    ARY3[0..10].to_s
  end

  measure.report("idx<<") do
    ARY1.each_with_index do |elem, idx|
      ARY3B << elem + ARY2[idx]
    end

    ARY3B[0..10].to_s
  end

  measure.report("map!") do
    ARY1B.map!.with_index do |elem, idx|
      elem + ARY2[idx]
    end

    ARY1B[0..10].to_s
  end

  measure.report("matrix") do
    (M1 + M2).to_a[0..10].to_s
  end

  measure.report("matrix 2d") do
    (M12D + M22D).to_a[0][0..10].to_s
  end

  measure.report("numo/narray") do
    (M1N1 + M2N1).to_a[0..10].to_s
  end

  measure.report("numo/narray 2d") do
    (M1N2 + M2N2).to_a[0..10].to_s
  end

  measure.report("bohrium ones add") do
    BH1.add(BH2).to_ary[0..10].to_s
  end

  measure.report("bohrium ones add!") do
    BH1.add!(BH2)
    BH1.to_ary[0..10].to_s
  end

  measure.report("bohrium ones +") do
    (BH1 + BH2).to_ary[0..10].to_s
  end
end
