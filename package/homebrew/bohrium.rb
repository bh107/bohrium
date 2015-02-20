require "formula"

class Bohrium < Formula
  homepage "http://bh107.org/"
  head "https://bitbucket.org/bohrium/bohrium.git"
  url "https://bitbucket.org/bohrium/bohrium/get/2bb557d48f8b485ef99d4b89b17989f5a7cc963f.zip"
  version "v0.2-1724-g2bb557d"
  sha1 "624549bf072ec3054780365d2ecfa44ed75307f7"

  depends_on "cmake" => :build
  depends_on "mono" => :build
  depends_on "swig" => :build
  depends_on "Python" => :build
  depends_on "boost" => [:build,  "universal"]
  #depends_on "cheetah" => :build

  head do
    url "https://bitbucket.org/bohrium/bohrium.git"
  end

  def install
    if build.head?
      ln_s cached_download/".git", ".git"
    end

    system "cmake", ".", *std_cmake_args
    system "make", "install"
    system "touch", "#{prefix}/var/bohrium/objects/.empty"
    system "touch", "#{prefix}/var/bohrium/kernels/.empty"
  end

  test do
    system "test/c/helloworld/bh_hello_world_c"
  end
end
