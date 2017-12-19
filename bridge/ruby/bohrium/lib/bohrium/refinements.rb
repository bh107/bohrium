module BohriumExtensions
  # Refine op+, such that if called with BhArray, we rearrange the operands so addition still works
  # e.g.
  #   BhArray.new([1, 2, 3]) + 2
  # is the same as
  #   2 + BhArray.new([1, 2, 3])
  def +(other)
    case other
    when BhArray
      # a + b == b + a
      other + self
    else
      super(other)
    end
  end

  def -(other)
    case other
    when BhArray
      # a - b = -b + a
      (other * -1) + self
    else
      super(other)
    end
  end

  def *(other)
    case other
    when BhArray
      # a * b = b * a
      other * self
    else
      super(other)
    end
  end
end

module BohriumFloatExtensions
  def /(other)
    case other
    when BhArray
      # a / b = (b / a) ** -1
      (other / self) ** -1
    else
      super(other)
    end
  end
end

class Integer
  prepend BohriumExtensions
end

class Float
  prepend BohriumExtensions
  prepend BohriumFloatExtensions
end

class Boolean
  prepend BohriumExtensions
end
