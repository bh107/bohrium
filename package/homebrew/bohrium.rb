require "formula"

class Bohrium < Formula
  homepage "http://bh107.org/"
  head "https://bitbucket.org/bohrium/bohrium.git"
  url "https://bitbucket.org/bohrium/bohrium/get/763e92f6f1db529bb69753f971b92599c7d982d0.zip"
  version "v0.2-1716-g763e92f"
  sha1 "857fba16e650bc549d7d1266dba4dcecf7628666"

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
