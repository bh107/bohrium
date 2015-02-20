require "formula"

class Bohrium < Formula
  homepage "http://bh107.org/"
  head "https://bitbucket.org/bohrium/bohrium.git"
  url "https://bitbucket.org/bohrium/bohrium/get/02b3ebf9c6bd53fd00d150ccdaaf9f5e77aee8dd.zip"
  version "v0.2-1709-g02b3ebf"
  sha1 "5c463ed46f6a4e9b65f95526c324b8641143a485"

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
