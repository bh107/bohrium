require "bohrium"

describe BhArray do
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

    context "given an array and a constant" do
      before do
        @a = BhArray.new([1, 2, 3])
      end

      it "finds the sum" do
        expect(@a.add(1).to_ary).to eq([2, 3, 4])
      end

      it "finds the sum using op+" do
        expect((@a + 1).to_ary).to eq([2, 3, 4])
      end
    end

    context "given two arrays created differently" do
      before do
        @a = BhArray.new([1, 2, 3])
        @b = BhArray.ones(3, 1, 1)
      end

      it "finds the sum" do
        expect((@a + @b).to_ary).to eq([2, 3, 4])
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

    context "given an array and a constant" do
      before do
        @a = BhArray.new([1, 2, 3])
      end

      it "finds the difference" do
        expect(@a.subtract(1).to_ary).to eq([0, 1, 2])
      end

      it "finds the difference using op-" do
        expect((@a - 1).to_ary).to eq([0, 1, 2])
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

    context "given an array and a constant" do
      before do
        @a = BhArray.new([1, 2, 3])
      end

      it "finds the product" do
        expect(@a.multiply(2).to_ary).to eq([2, 4, 6])
      end

      it "finds the product using op*" do
        expect((@a * 2).to_ary).to eq([2, 4, 6])
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

    context "given an array and a constant" do
      before do
        @a = BhArray.new([2, 4, 6])
      end

      it "finds the quotient" do
        expect(@a.divide(2).to_ary).to eq([1, 2, 3])
      end

      it "finds the quotient using op/" do
        expect((@a / 2).to_ary).to eq([1, 2, 3])
      end
    end
  end

  describe "#power" do
    context "given two arrays" do
      before do
        @a = BhArray.new([2, 2])
        @b = BhArray.new([2, 3])
      end

      it "finds the power" do
        expect(@a.power(@b).to_ary).to eq([4, 8])
      end

      it "find the power using op**" do
        expect((@a ** @b).to_ary).to eq([4, 8])
      end
    end

    context "given two floating point arrays" do
      before do
        @a = BhArray.new([1.0, 3.3])
        @b = BhArray.new([0.5, 2.0])
      end

      it "finds the power" do
        expect(@a.power(@b).to_ary[0]).to be_within(0.001).of(1.0)
        expect(@a.power(@b).to_ary[1]).to be_within(0.001).of(10.8899)
      end
    end

    context "given an array and a constant" do
      before do
        @a = BhArray.new([2, 4, 6])
      end

      it "finds the power" do
        expect(@a.power(2).to_ary).to eq([4, 16, 36])
      end

      it "finds the power using op**" do
        expect((@a ** 2).to_ary).to eq([4, 16, 36])
      end
    end
  end
end
