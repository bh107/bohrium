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

  describe "#add" do
    context "given two arrays" do
      before do
        @a = BhArray.new([1, 1])
        @b = BhArray.new([2, 2])
      end

      it "finds the sum" do
        expect(@a.add(@b).to_ary).to eq([3, 3])
      end

      it "find the sum using op+" do
        expect((@a + @b).to_ary).to eq([3, 3])
      end
    end

    context "given two floating point arrays" do
      before do
        @a = BhArray.new([5.5, 2.3])
        @b = BhArray.new([4.5, 7.7])
      end

      it "finds the sum" do
        expect(@a.add(@b).to_ary).to eq([10.0, 10.0])
      end
    end
  end

  describe "#subtract" do
    context "given two arrays" do
      before do
        @a = BhArray.new([4, 4])
        @b = BhArray.new([1, 1])
      end

      it "finds the difference" do
        expect(@a.subtract(@b).to_ary).to eq([3, 3])
      end

      it "find the difference using op-" do
        expect((@a - @b).to_ary).to eq([3, 3])
      end
    end

    context "given two floating point arrays" do
      before do
        @a = BhArray.new([5.5, 2.3])
        @b = BhArray.new([4.5, 2.3])
      end

      it "finds the difference" do
        expect(@a.subtract(@b).to_ary).to eq([1.0, 0.0])
      end
    end
  end

  describe "#multiply" do
    context "given two arrays" do
      before do
        @a = BhArray.new([4, 4])
        @b = BhArray.new([2, 4])
      end

      it "finds the product" do
        expect(@a.multiply(@b).to_ary).to eq([8, 16])
      end

      it "find the product using op*" do
        expect((@a * @b).to_ary).to eq([8, 16])
      end
    end

    context "given two floating point arrays" do
      before do
        @a = BhArray.new([1.5, 2.5])
        @b = BhArray.new([1.5, 6.2])
      end

      it "finds the product" do
        expect(@a.multiply(@b).to_ary).to eq([2.25, 15.5])
      end
    end
  end

  describe "#divide" do
    context "given two arrays" do
      before do
        @a = BhArray.new([6, 6])
        @b = BhArray.new([2, 6])
      end

      it "finds the quotient" do
        expect(@a.divide(@b).to_ary).to eq([3, 1])
      end

      it "find the quotient using op/" do
        expect((@a / @b).to_ary).to eq([3, 1])
      end
    end

    context "given two floating point arrays" do
      before do
        @a = BhArray.new([1.0, 2.5])
        @b = BhArray.new([0.5, 2.5])
      end

      it "finds the quotient" do
        expect(@a.divide(@b).to_ary).to eq([2.0, 1.0])
      end
    end
  end
end
