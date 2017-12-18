require "bohrium"

describe BhArray do
  context "given an array" do
    before do
      @pi = BhArray.new([Math::PI])
      @zero = BhArray.new([0.0])
    end

    {
      "cos"     => "cos",
      "sin"     => "sin",
      "tan"     => "tan",
      "cosh"    => "cosh",
      "sinh"    => "sinh",
      "tanh"    => "tanh",
      "arccosh" => "acosh",
      "arcsinh" => "asinh",
    }.each do |bhop, rbop|
      it "finds #{bhop} of the PI" do
        expect(@pi.send(bhop.to_sym).to_a[0]).to be_within(0.1).of(Math.send(rbop.to_sym, Math::PI))
      end
    end

    {
      "arccos"  => "acos",
      "arcsin"  => "asin",
      "arctan"  => "atan",
      "arctanh" => "atanh"
    }.each do |bhop, rbop|
      it "finds #{bhop} of the 1" do
        expect(@zero.send(bhop.to_sym).to_a[0]).to be_within(0.1).of(Math.send(rbop.to_sym, 0.0))
      end
    end

    it "finds e** for the array" do
      expect(@pi.exp.to_a[0]).to be_within(0.1).of(Math.exp(Math::PI))
    end

    it "finds 2** for the array" do
      expect(@pi.exp2.to_a[0]).to be_within(0.1).of(2**Math::PI)
    end
  end
end
