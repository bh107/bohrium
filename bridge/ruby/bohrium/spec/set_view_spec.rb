require "bohrium"

describe BhArray do
  describe "#[]" do
    context "given an array" do
      before do
        @a = BhArray.arange(6)
      end

      it "return the values of the view after changing it" do
        expect(@a.shape).to eq([6])
        @a[true] = 666
        expect(@a.to_ary).to eq([666, 666, 666, 666, 666, 666])
      end
    end

    context "given a 2D array" do
      before do
        @a = BhArray.arange(9).reshape([3, 3])
      end

      it "return the values after changing the view from 0..1, 0..1" do
        expect(@a[0..1, 0..1].shape).to eq([2, 2])
        @a[0..1, 0..1] = BhArray.new([4, 4, 4, 4]).reshape([2, 2])
        expect(@a.to_ary).to eq([[ 4,  4,  2],
                                 [ 4,  4,  5],
                                 [ 6,  7,  8]])
      end

      it "return the values after changing the view from 1..2, 1..2" do
        expect(@a[1..2, 1..2].shape).to eq([2, 2])
        @a[1..2, 1..2] = BhArray.new([4, 4, 4, 4]).reshape([2, 2])
        expect(@a.to_ary).to eq([[ 0, 1, 2],
                                 [ 3, 4, 4],
                                 [ 6, 4, 4]])
      end

      it "return the values after changing the view from 0..1, 1..2" do
        expect(@a[0..1, 1..2].shape).to eq([2, 2])
        @a[0..1, 1..2] = BhArray.new([4, 4, 4, 4]).reshape([2, 2])
        expect(@a.to_ary).to eq([[ 0, 4, 4],
                                 [ 3, 4, 4],
                                 [ 6, 7, 8]])
      end

      it "return the values after changing the view from 1..2, 0..1" do
        expect(@a[1..2, 0..1].shape).to eq([2, 2])
        @a[1..2, 0..1] = BhArray.new([4, 4, 4, 4]).reshape([2, 2])
        expect(@a.to_ary).to eq([[ 0, 1, 2],
                                 [ 4, 4, 5],
                                 [ 4, 4, 8]])
      end

      it "return the values after changing the view from 0..0, 1..2" do
        expect(@a[0..0, 1..2].shape).to eq([1, 2])
        @a[0..0, 1..2] = BhArray.new([4, 4]).reshape([1, 2])
        expect(@a.to_ary).to eq([[ 0, 4, 4],
                                 [ 3, 4, 5],
                                 [ 6, 7, 8]])
      end

      it "return the values after changing the view from 1..2, 0..0" do
        expect(@a[1..2, 0..0].shape).to eq([2, 1])
        @a[1..2, 0..0] = BhArray.new([4, 4]).reshape([2, 1])
        expect(@a.to_ary).to eq([[ 0, 1, 2],
                                 [ 4, 4, 5],
                                 [ 4, 7, 8]])
      end

      it "return the values after changing the view from 1..2, true" do
        expect(@a[1..2, true].shape).to eq([2, 3])
        @a[1..2, true] = BhArray.new([4, 4, 4, 4, 4, 4]).reshape([2, 3])
        expect(@a.to_ary).to eq([[ 0, 1, 2],
                                 [ 4, 4, 4],
                                 [ 4, 4, 4]])
      end

      it "return the values after changing the view from true, 1..2" do
        expect(@a[true, 1..2].shape).to eq([3, 2])
        @a[true, 1..2] = BhArray.new([4, 4, 4, 4, 4, 4]).reshape([3, 2])
        expect(@a.to_ary).to eq([[ 0, 4, 4],
                                 [ 3, 4, 4],
                                 [ 6, 4, 4]])
      end
    end
  end
end
