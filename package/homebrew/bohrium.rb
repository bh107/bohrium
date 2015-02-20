require "formula"

class Bohrium < Formula
  homepage "http://bh107.org/"
  head "https://bitbucket.org/bohrium/bohrium.git"
  url "https://bitbucket.org/bohrium/bohrium/get/c1fbbca3a21abfaf95e03d2b7bb64a5ab4cc273f.zip"
  version "v0.2-1713-gc1fbbca"
  sha1 "1e9f8bac7b109c4b8b4e2a095f60c918fd2363b4"

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
    end

    system "cmake", ".", *std_cmake_args
    system "make", "install"
  end

  test do
    system "test/c/helloworld/bh_hello_world_c"
  end
end
