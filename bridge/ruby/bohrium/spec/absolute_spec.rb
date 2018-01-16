require "bohrium"

describe BhArray do
  describe "#absolute" do
    context "given an array with some negative elements" do
      before do
        @a = BhArray.new([1, 2, -3, -4, 5])
      end

      it "return the absolute values" do
        expect(@a.absolute.to_ary).to eq([1, 2, 3, 4, 5])
      end
    end

    context "given a 2D array with some negative elements" do
      before do
        @a = (BhArray.arange(9)-3).reshape([3, 3])
      end

      it "return the absolute values" do
        expect(@a.absolute.to_ary).to eq([[3, 2, 1], [0, 1, 2], [3, 4, 5]])
      end
    end
  end
end
