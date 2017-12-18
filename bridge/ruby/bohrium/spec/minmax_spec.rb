require "bohrium"

describe BhArray do
  describe "#minimum" do
    context "given two arrays" do
      before do
        @a = BhArray.new([1, 2, 3])
        @b = BhArray.new([3, 2, 1])
      end

      it "finds the elementwise minimum" do
        expect(@a.minimum(@b).to_ary).to eq([1, 2, 1])
      end
    end
  end

  describe "#maximum" do
    context "given two arrays" do
      before do
        @a = BhArray.new([1, 2, 3])
        @b = BhArray.new([3, 2, 1])
      end

      it "finds the elementwise maximum" do
        expect(@a.maximum(@b).to_ary).to eq([3, 2, 3])
      end
    end
  end
end
