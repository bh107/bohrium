require "benchmark"

n = 100_000_000
puts "COS NO INIT"
puts "n = #{n}"

ARY1 = [Math::PI] * n

require "numo/narray"
NUMO1 = Numo::DFloat.new(n).fill(Math::PI)

require "bohrium"
BH1 = BhArray.ones(n, 1, Math::PI)

Benchmark.bm(20) do |measure|
  measure.report("ruby") do
    ARY1.map { |e| Math::cos(e) }[0..10].to_s
  end

  measure.report("numo/narray") do
    Numo::DFloat::Math.cos(NUMO1).to_a[0..10].to_s
  end

  measure.report("bohrium") do
    BH1.cos.to_a[0..10].to_s
  end
end
