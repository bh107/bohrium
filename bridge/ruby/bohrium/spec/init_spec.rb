require "bohrium"

describe BhArray do
  describe "#initialize" do
    context "given an array" do
      before do
        @a = BhArray.new([1, 2, 3])
      end

      it "contains that array" do
        expect(@a.to_ary).to eq([1, 2, 3])
      end
    end
  end

  describe "#ones" do
    context "given a BhArray" do
      before do
        @a = BhArray.ones(10)
      end

      it "contains an array of same size" do
        expect(@a.to_ary.size).to eq(10)
      end

      it "contains an array with only ones" do
        expect(@a.to_ary.uniq).to eq([1])
      end
    end

    context "given a two dimensional array" do
      before do
        @a = BhArray.ones(10, 10)
      end

      it "contains the product of dimensions elements" do
        expect(@a.to_ary.size).to eq(100)
      end

      it "contains only ones" do
        expect(@a.to_ary.uniq).to eq([1])
      end
    end

    context "given a value" do
      before do
        @a = BhArray.ones(10, 1, 7)
      end

      it "contains an array of same size" do
        expect(@a.to_ary.size).to eq(10)
      end

      it "contains only that value" do
        expect(@a.to_ary.uniq).to eq([7])
      end
    end

    context "given a float" do
      before do
        @a = BhArray.ones(10, 1, 5.5)
      end

      it "contains only that value" do
        expect(@a.to_ary.uniq).to eq([5.5])
      end
    end
  end

  describe "#zeros" do
    context "given a BhArray" do
      before do
        @a = BhArray.zeros(10)
      end

      it "contains an array of same size" do
        expect(@a.to_ary.size).to eq(10)
      end

      it "contains an array with only zeros" do
        expect(@a.to_ary.uniq).to eq([0])
      end
    end

    context "given a two dimensional array" do
      before do
        @a = BhArray.zeros(10, 10)
      end

      it "contains the product of dimensions elements" do
        expect(@a.to_ary.size).to eq(100)
      end

      it "contains only zeros" do
        expect(@a.to_ary.uniq).to eq([0])
      end
    end
  end
end
