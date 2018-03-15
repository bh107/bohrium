#   FREE ???
# BH_ADD a744[0:10:10,0:10:1] a1[1:11:12,1:11:1] a1[0:10:12,1:11:1]
# BH_ADD a745[0:10:10,0:10:1] a744[0:10:10,0:10:1] a1[2:11:12,1:11:1]
#   FREE a744
# BH_ADD a746[0:10:10,0:10:1] a745[0:10:10,0:10:1] a1[1:11:12,0:10:1]
#   FREE a745
# BH_ADD a747[0:10:10,0:10:1] a746[0:10:10,0:10:1] a1[1:11:12,2:11:1]
#   FREE a746
# BH_MULTIPLY a748[0:10:10,0:10:1] a747[0:10:10,0:10:1] 2.00000002980232239e-01f
#   FREE a747
#   FREE ???
# BH_SUBTRACT a749[0:10:10,0:10:1] a748[0:10:10,0:10:1] a1[1:11:12,1:11:1]
# BH_ABSOLUTE a750[0:10:10,0:10:1] a749[0:10:10,0:10:1]
#   FREE a749
# BH_ADD_REDUCE a751[0:10:1] a750[0:10:10,0:10:1] 0
# BH_ADD_REDUCE a752[0:1:1] a751[0:10:1] 0
#   FREE a751
#   FREE a750
#   FREE ???
# BH_IDENTITY a1[1:11:12,1:11:1] a748[0:10:10,0:10:1]
# BH_GREATER a743[0:1:1] a752[0:1:1] 4.20000000000000000e+01f

require "bohrium"

def heat2d(height, width, epsilon=42)
  grid = BhArray.ones(height+2, width+2, 0.0) # Force 'float' type
  grid[    true,       0] = -273.15
  grid[    true, width+1] = -273.15
  grid[height+1,    true] = -273.15
  grid[       0,    true] = 40.0

  center = grid[1..height,   1..width]
  north  = grid[0..height-1, 1..width]
  south  = grid[2..height,   1..width]
  east   = grid[1..height,   0..width-1]
  west   = grid[1..height,   2..width]

  delta = BhArray.ones(1, 1, epsilon+1)
  while (delta > epsilon).to_ary[0]
    tmp = 0.2 * (center + north + south + east + west)
    delta = (tmp-center).absolute.add_reduce(0).add_reduce(0)
    grid[1..height, 1..width] = tmp

    Bohrium.flush
  end

  center
end

def mprint(ary)
  print "["

  ary.each_with_index do |row, idx|
    if idx.zero?
      print "["
    else
      print " ["
    end

    print row.map { |f| f.round(8).to_s.rjust(13) }.join(", ")

    if idx == row.size-1
      print "]"
    else
      puts "],"
    end
  end

  puts "]"
end

size = ARGV[0].to_i
result = heat2d(size, size)
puts result.to_ary[5][5]
#mprint result.to_ary
