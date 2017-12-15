require "benchmark"

n = 100_000_000
puts "ADD"
puts "n = #{n}"

Benchmark.bm(20) do |measure|
  measure.report("zip") do
    output = ""

    ary1 = [2] * n
    ary2 = [3] * n

    output << ary1.zip(ary2).map { |i| i.reduce(:+) }[0..10].to_s
  end

  measure.report("zip_lazy") do
    output = ""

    ary1 = [2] * n
    ary2 = [3] * n

    output << ary1.lazy.zip(ary2).map { |i| i.reduce(:+) }.to_a[0..10].to_s
  end

  measure.report("zip_sum") do
    output = ""

    ary1 = [2] * n
    ary2 = [3] * n

    output << ary1.zip(ary2).map(&:sum)[0..10].to_s
  end

  measure.report("zip2") do
    output = ""

    ary1 = [2] * n
    ary2 = [3] * n

    output << ary1.zip(ary2).map { |a, b| a + b }[0..10].to_s
  end

  measure.report("idx") do
    output = ""

    ary1 = [2] * n
    ary2 = [3] * n
    ary3 = Array.new(n)

    ary1.each_with_index do |elem, idx|
      ary3[idx] = elem + ary2[idx]
    end

    output << ary3[0..10].to_s
  end

  measure.report("idx<<") do
    output = ""

    ary1 = [2] * n
    ary2 = [3] * n
    ary3 = []

    ary1.each_with_index do |elem, idx|
      ary3 << elem + ary2[idx]
    end

    output << ary3[0..10].to_s
  end

  measure.report("map!") do
    output = ""

    ary1 = [2] * n
    ary2 = [3] * n

    ary1.map!.with_index do |elem, idx|
      elem + ary2[idx]
    end

    output << ary1[0..10].to_s
  end

  measure.report("matrix") do
    output = ""

    require "matrix"
    m1 = Matrix[[2] * n]
    m2 = Matrix[[3] * n]
    output << (m1 + m2).to_a[0..10].to_s
  end

  measure.report("matrix 2d") do
    output = ""

    require "matrix"
    m1 = Matrix[[[2] * n]]
    m2 = Matrix[[[3] * n]]
    output << (m1 + m2).to_a[0][0..10].to_s
  end

  measure.report("numo/narray") do
    output = ""

    require "numo/narray"
    m1 = Numo::Int64.new(n).fill(2)
    m2 = Numo::Int64.new(n).fill(3)
    output << (m1 + m2).to_a[0..10].to_s
  end

  measure.report("numo/narray 2d") do
    output = ""

    require "numo/narray"
    m1 = Numo::Int64.new(n, 1).fill(2)
    m2 = Numo::Int64.new(n, 1).fill(3)
    output << (m1 + m2).to_a[0..10].to_s
  end

  measure.report("bohrium") do
    output = ""

    require "bohrium"
    ary1 = [2] * n
    ary2 = [3] * n

    a = BhArray.new(ary1)
    b = BhArray.new(ary2)

    output << a.add(b).to_ary[0..10].to_s # flushes
  end

  measure.report("bohrium ones add") do
    output = ""

    require "bohrium"
    a = BhArray.ones(n, 1, 2)
    b = BhArray.ones(n, 1, 3)
    output << a.add(b).to_ary[0..10].to_s # flushes
  end

  measure.report("bohrium ones add!") do
    output = ""

    require "bohrium"
    a = BhArray.ones(n, 1, 2)
    b = BhArray.ones(n, 1, 3)
    a.add!(b)
    output << a.to_ary[0..10].to_s # flushes
  end

  measure.report("bohrium ones +") do
    output = ""

    require "bohrium"
    a = BhArray.ones(n, 1, 2)
    b = BhArray.ones(n, 1, 3)
    output << (a + b).to_ary[0..10].to_s
  end
end
