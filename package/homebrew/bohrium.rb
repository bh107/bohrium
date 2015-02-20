require "formula"

class Bohrium < Formula
  homepage "http://bh107.org/"
  head "https://bitbucket.org/bohrium/bohrium.git"
  url "https://bitbucket.org/bohrium/bohrium/get/f28bbf8c146a1f679cc9a3ef81f044e99f2d52d8.zip"
  version "v0.2-1712-gf28bbf8"
  sha1 "82cf69d85a6e054b1a60ec8b53b5a805e59c6fbb"

  depends_on "cmake" => :build
  depends_on "mono" => :build
  depends_on "swig" => :build
  depends_on "Python" => :build
  #depends_on "cheetah" => :build

  head do
    url "https://bitbucket.org/bohrium/bohrium.git"
  end

  def install
    if build.head?
      ln_s cached_download/".git", ".git"
      system "./bootstrap"
    end

    system "cmake", ".", *std_cmake_args
    system "make", "install"
  end

  test do
    system "test/c/helloworld/bh_hello_world_c"
  end
end
