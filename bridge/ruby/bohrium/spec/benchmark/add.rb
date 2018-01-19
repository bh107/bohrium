require "benchmark"
require "csv"

class ::Float
  def to_s
    "%.10f" % self
  end
end

headers = []
max = (ARGV[0] || 4).to_i
filename = ARGV[1] || "add_result.csv"
csv_result = File.open(filename, "w+")

CSV(csv_result) do |csv_str|
  (0..max).each do |i|
    csv = []
    n = (10**i).to_i
    puts "ADD: n = #{n} (10e#{i})"

    Benchmark.bm(20) do |measure|
      m = measure.report("zip") do
        output = ""

        ary1 = [2] * n
        ary2 = [3] * n

        output << ary1.zip(ary2).map { |i| i.reduce(:+) }[0..10].to_s
      end
      headers << m.label if i.zero?
      csv << m.real.round(8)


      m = measure.report("zip_lazy") do
        output = ""

        ary1 = [2] * n
        ary2 = [3] * n

        output << ary1.lazy.zip(ary2).map { |i| i.reduce(:+) }.to_a[0..10].to_s
      end
      headers << m.label if i.zero?
      csv << m.real.round(8)


      m = measure.report("zip_sum") do
        output = ""

        ary1 = [2] * n
        ary2 = [3] * n

        output << ary1.zip(ary2).map(&:sum)[0..10].to_s
      end
      headers << m.label if i.zero?
      csv << m.real.round(8)


      m = measure.report("zip2") do
        output = ""

        ary1 = [2] * n
        ary2 = [3] * n

        output << ary1.zip(ary2).map { |a, b| a + b }[0..10].to_s
      end
      headers << m.label if i.zero?
      csv << m.real.round(8)


      m = measure.report("idx") do
        output = ""

        ary1 = [2] * n
        ary2 = [3] * n
        ary3 = Array.new(n)

        ary1.each_with_index do |elem, idx|
          ary3[idx] = elem + ary2[idx]
        end

        output << ary3[0..10].to_s
      end
      headers << m.label if i.zero?
      csv << m.real.round(8)


      m = measure.report("idx<<") do
        output = ""

        ary1 = [2] * n
        ary2 = [3] * n
        ary3 = []

        ary1.each_with_index do |elem, idx|
          ary3 << elem + ary2[idx]
        end

        output << ary3[0..10].to_s
      end
      headers << m.label if i.zero?
      csv << m.real.round(8)


      m = measure.report("map!") do
        output = ""

        ary1 = [2] * n
        ary2 = [3] * n

        ary1.map!.with_index do |elem, idx|
          elem + ary2[idx]
        end

        output << ary1[0..10].to_s
      end
      headers << m.label if i.zero?
      csv << m.real.round(8)


      m = measure.report("matrix") do
        output = ""

        require "matrix"
        m1 = Matrix[[2] * n]
        m2 = Matrix[[3] * n]
        output << (m1 + m2).to_a[0..10].to_s
      end
      headers << m.label if i.zero?
      csv << m.real.round(8)


      m = measure.report("matrix 2d") do
        output = ""

        require "matrix"
        m1 = Matrix[[[2] * n]]
        m2 = Matrix[[[3] * n]]
        output << (m1 + m2).to_a[0][0..10].to_s
      end
      headers << m.label if i.zero?
      csv << m.real.round(8)


      m = measure.report("numo/narray") do
        output = ""

        require "numo/narray"
        m1 = Numo::Int64.new(n).fill(2)
        m2 = Numo::Int64.new(n).fill(3)
        output << (m1 + m2).to_a[0..10].to_s
      end
      headers << m.label if i.zero?
      csv << m.real.round(8)


      m = measure.report("numo/narray 2d") do
        output = ""

        require "numo/narray"
        m1 = Numo::Int64.new(n, 1).fill(2)
        m2 = Numo::Int64.new(n, 1).fill(3)
        output << (m1 + m2).to_a[0..10].to_s
      end
      headers << m.label if i.zero?
      csv << m.real.round(8)


      m = measure.report("numo/narray 2d inverse") do
        output = ""

        require "numo/narray"
        m1 = Numo::Int64.new(1, n).fill(2)
        m2 = Numo::Int64.new(1, n).fill(3)
        output << (m1 + m2).to_a[0..10].to_s
      end
      headers << m.label if i.zero?
      csv << m.real.round(8)


      m = measure.report("bohrium") do
        output = ""

        require "bohrium"
        ary1 = [2] * n
        ary2 = [3] * n

        a = BhArray.new(ary1)
        b = BhArray.new(ary2)

        output << a.add(b).to_ary[0..10].to_s # flushes
      end
      headers << m.label if i.zero?
      csv << m.real.round(8)


      m = measure.report("bohrium ones add") do
        output = ""

        require "bohrium"
        a = BhArray.ones(1, n, 2)
        b = BhArray.ones(1, n, 3)
        output << a.add(b).to_ary[0..10].to_s # flushes
      end
      headers << m.label if i.zero?
      csv << m.real.round(8)


      m = measure.report("bohrium ones add!") do
        output = ""

        require "bohrium"
        a = BhArray.ones(1, n, 2)
        b = BhArray.ones(1, n, 3)
        a.add!(b)
        output << a.to_ary[0..10].to_s # flushes
      end
      headers << m.label if i.zero?
      csv << m.real.round(8)


      m = measure.report("bohrium ones +") do
        output = ""

        require "bohrium"
        a = BhArray.ones(1, n, 2)
        b = BhArray.ones(1, n, 3)
        output << (a + b).to_ary[0..10].to_s
      end
      headers << m.label if i.zero?
      csv << m.real.round(8)

      csv_str << headers if i.zero?
      csv_str << csv

      puts "\n\n==========\n\n"
    end
  end
end

puts "== DONE =="
