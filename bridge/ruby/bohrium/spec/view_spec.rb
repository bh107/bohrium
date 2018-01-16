require "bohrium"

describe BhArray do
  describe "#[]" do
    context "given an array" do
      before do
        @a = BhArray.arange(6)
      end

      it "return the values of the view" do
        expect(@a.shape).to eq([1, 6])
        expect(@a.to_ary).to eq([0, 1, 2, 3, 4, 5])
      end
    end

    context "given a 2D array" do
      before do
        @a = BhArray.arange(25).reshape([5, 5])
      end

      it "return the values" do
        expect(@a.shape).to eq([5, 5])
        expect(@a.to_ary).to eq([[ 0,  1,  2,  3,  4],
                                 [ 5,  6,  7,  8,  9],
                                 [10, 11, 12, 13, 14],
                                 [15, 16, 17, 18, 19],
                                 [20, 21, 22, 23, 24]])
      end

      it "return the values of the view from 0..1, 0..1" do
        expect(@a[0..1, 0..1].shape).to eq([2, 2])
        expect(@a[0..1, 0..1].to_ary).to eq([[ 0,  1],
                                             [ 5,  6]])
      end

      it "return the values of the view from 1..3, 1..3" do
        expect(@a[1..3, 1..3].shape).to eq([3, 3])
        expect(@a[1..3, 1..3].to_ary).to eq([[ 6,  7,  8],
                                             [11, 12, 13],
                                             [16, 17, 18]])
      end

      it "return the values of the view from 0..1, 1..3" do
        expect(@a[0..1, 1..3].shape).to eq([2, 3])
        expect(@a[0..1, 1..3].to_ary).to eq([[1, 2, 3],
                                             [6, 7, 8]])
      end

      it "return the values of the view from 1..3, 0..1" do
        expect(@a[1..3, 0..1].shape).to eq([3, 2])
        expect(@a[1..3, 0..1].to_ary).to eq([[ 5,  6],
                                             [10, 11],
                                             [15, 16]])
      end

      it "return the values of the view from 0..0, 1..3" do
        expect(@a[0..0, 1..3].shape).to eq([1, 3])
        expect(@a[0..0, 1..3].to_ary).to eq([1, 2, 3])
      end

      it "return the values of the view from 1..3, 0..0" do
        expect(@a[1..3, 0..0].shape).to eq([3, 1])
        expect(@a[1..3, 0..0].to_ary).to eq([[5], [10], [15]])
      end

      it "return the values of the view from true, true" do
        expect(@a[true, true].shape).to eq([5, 5])
        expect(@a[true, true].to_ary).to eq(@a.to_ary)
      end

      it "return the values of the view from true, 0..1" do
        expect(@a[true, 0..1].shape).to eq([5, 2])
        expect(@a[true, 0..1].to_ary).to eq([[ 0,  1],
                                             [ 5,  6],
                                             [10, 11],
                                             [15, 16],
                                             [20, 21]])
      end

      it "return the values of the view from 0..1, true" do
        expect(@a[0..1, true].shape).to eq([2, 5])
        expect(@a[0..1, true].to_ary).to eq([[ 0,  1, 2, 3, 4],
                                             [ 5,  6, 7, 8, 9]])
      end
    end
  end
end
