require "benchmark"
require "csv"

class ::Float
  def to_s
    "%.10f" % self
  end
end

headers = []
rows = (ARGV[0] || 4).to_i
filename = ARGV[1] || "add_reduce.csv"
csv_result = File.open(filename, "w+")

CSV(csv_result) do |csv_str|
  (1..rows).each do |row|
    csv = []
    r = (10**row).to_i
    c = (2000).to_i
    puts "ADD_REDUCE columns: r*c = #{r}*#{c} (10**#{row} * 2000)"

    Benchmark.bm(20) do |measure|
      m = measure.report("vanilla") do
        output = ""

        ary = []
        i = 0
        r.times do
          ary << []
          c.times do
            ary[-1] << i
            i += 1
          end
        end

        result = Array.new(c) { 0 }
        c.times do |col_idx|
          r.times do |row_idx|
            result[col_idx] += ary[row_idx][col_idx]
          end
        end

        output << result[0..10].to_s
      end

      headers << m.label if row.zero?
      csv << m.real.round(8)


      m = measure.report("numo/narray") do
        output = ""

        require "numo/narray"
        ary = Numo::Int32.new(r * c).seq.reshape(r, c).sum(axis: 0)

        output << ary.to_a[0..10].to_s
      end
      headers << m.label if row.zero?
      csv << m.real.round(8)


      m = measure.report("bohrium") do
        output = ""

        require "bohrium"
        ary = BhArray.arange(r * c).reshape([r, c]).add_reduce(0)

        output << ary.to_ary[0..10].to_s # flushes
      end
      headers << m.label if row.zero?
      csv << m.real.round(8)

      puts "\n\n==========\n\n"
    end
  end
end

puts "== DONE =="
