require "bohrium"

describe BhArray do
  context "given an integer array" do
    before do
      @b = BhArray.new([1, 2, 3])
    end

    it "finds the sum with constant" do
      expect((2 + @b).to_a).to eq([3, 4, 5])
      expect((2 + @b).to_a).to eq((@b + 2).to_a) # a + b = b + a
    end

    it "finds the difference with constant" do
      expect((2 - @b).to_a).to eq([1, 0, -1])
      expect((2 - @b).to_a).to eq(((@b * -1) + 2).to_a) # a - b = -b + a = b*-1 + a
    end

    it "finds the product with constant" do
      expect((2 * @b).to_a).to eq([2, 4, 6])
      expect((2 * @b).to_a).to eq((@b * 2).to_a) # a * b = b * a
    end
  end

  context "given a float array" do
    before do
      @b = BhArray.new([1.0, 2.0, 3.0])
    end

    it "finds the sum with constant" do
      expect((2.0 + @b).to_a).to eq([3.0, 4.0, 5.0])
      expect((2.0 + @b).to_a).to eq((@b + 2.0).to_a) # a + b = b + a
    end

    it "finds the difference with constant" do
      expect((2.0 - @b).to_a).to eq([1.0, 0.0, -1.0])
      expect((2.0 - @b).to_a).to eq(((@b * -1.0) + 2.0).to_a) # a - b = -b + a = b*-1 + a
    end

    it "finds the product with constant" do
      expect((2.0 * @b).to_a).to eq([2.0, 4.0, 6.0])
      expect((2.0 * @b).to_a).to eq((@b * 2.0).to_a) # a * b = b * a
    end

    it "finds the quotient with constant" do
      expect((6.0 / @b).to_a).to eq([6.0, 3.0, 2.0])
      expect((6.0 / @b).to_a).to eq(((@b / 6.0) ** -1).to_a) # a / b = (b / a) ** -1
    end
  end
end
