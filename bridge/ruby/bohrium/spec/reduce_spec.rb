require "bohrium"
require "pp"

describe BhArray do
  context "given an identity" do
    before do
      @a = BhArray.ones(3, 2, 3)
    end

    it "finds the sum" do
      expect(@a.add_reduce(1).to_ary).to eq([6, 6, 6])
    end
  end

  context "given an integer array" do
    before do
      @a = BhArray.arange(7)
    end

    it "find the sum by reducing to one element" do
      expect(@a.add_reduce(0).to_ary).to eq([21])
    end
  end
end
