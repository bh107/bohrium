require "bohrium/version"
require "bohrium/bohrium.bundle"

require "bohrium/refinements"

class BhArray
  def [](*args)
    # Ensure that all the dimensions are accounted for.
    args.unshift(true) until args.size == self.shape.size

    ranges = convert_ranges(*args)
    view_from_ranges(*ranges)
  end

  def []=(*args, value)
    # Ensure that all the dimensions are accounted for.
    args.unshift(true) until args.size == self.shape.size

    ranges = convert_ranges(*args)
    # If the value is an array, convert it to a BhArray.
    value = BhArray.new(value) if Array === value
    set_from_ranges(*ranges, value)
  end

  private

  def convert_ranges(*args)
    args.map.with_index do |arg, idx|
      case arg
      when Integer
        # A fixnum is just a range from and to the same number.
        [arg, arg]
      when Range
        fail "Cannot index with anything but integers." unless arg.all? { |x| Integer === x }

        # If the end is excluded, remove it from the range.
        if arg.exclude_end?
          [arg.begin, arg.end-1]
        else
          [arg.begin, arg.end]
        end
      when TrueClass
        # If the argument is 'true', we return the entire dimension.
        [0, self.shape[idx]-1]
      when Symbol
        fail "Got ':#{arg}'. What now?"
      else
        fail "Subscript operator ([]) only works with ranges for now."
      end
    end
  end
end
