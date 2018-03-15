require "bohrium"

a = BhArray.ones(1, 5, 1)

5.times do |i|
  puts i
  a = BhArray.ones(1, 5, 1)
  a.print
end

a.print
