require "formula"

class Bohrium < Formula
  homepage "http://bh107.org/"
  head "https://bitbucket.org/bohrium/bohrium.git"
  url "https://bitbucket.org/bohrium/bohrium/get/4c5a34d1035db67e60c906cb07a4edbbd0fa16db.zip"
  version "v0.2-1718-g4c5a34d"
  sha1 "2d5a96ed0082fc3c56b34909e4c1eb851137f4e5"

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
    system "touch", "#{prefix}/var/bohrium/objects/.empty"
    system "touch", "#{prefix}/var/bohrium/kernels/.empty"
    system "touch", "#{prefix}/var/bohrium/fuse_cache/.empty"
  end

  test do
    system "test/c/helloworld/bh_hello_world_c"
  end
end
