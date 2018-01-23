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

  delta = epsilon+1
  while delta > epsilon
    tmp = 0.2 * (center + north + south + east + west)
    delta = (tmp-center).absolute.add_reduce(0).add_reduce(0).to_ary[0]
    grid[1..height, 1..width] = tmp
  end

  center
end

def mprint(ary)
  puts "["
  ary.each do |row|
    print "["
    print row.map { |f| f.round(3).to_s.rjust(8) }.join(", ")
    puts "]"
  end
  puts "]"
end

size = ARGV[0].to_i
result = heat2d(size, size)
result.to_ary
# mprint result.to_ary
