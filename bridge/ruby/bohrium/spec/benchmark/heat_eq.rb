require "bohrium"

def heat2d(height, width, epsilon=42)
  grid = BhArray.ones(height+2, width+2, 0.0) # Force 'float' type
  grid[    true,       0] = -273.15
  grid[    true, width+1] = -273.15
  grid[height+1,    true] = -273.15
  grid[       0,    true] = 40.0

  center = grid[1..height-1, 1..width-1]
  north  = grid[0..height-2, 1..width-1]
  south  = grid[2..height-1, 1..width-1]
  east   = grid[1..height-1, 0..width-2]
  west   = grid[1..height-1, 2..width-1]

  delta = epsilon+1
  while delta > epsilon
    tmp = 0.2 * (center + north + south + east + west)
    # delta = (tmp-center).absolute.add_reduce
    delta = (tmp-center).absolute.to_ary.flatten.reduce(:+)
    grid[1..height-1, 1..width-1] = tmp
  end

  center
end

result = heat2d(100, 100)
