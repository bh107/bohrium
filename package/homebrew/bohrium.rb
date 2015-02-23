require "formula"

class Bohrium < Formula
  homepage "http://bh107.org/"
  head "https://bitbucket.org/bohrium/bohrium.git"
  url "https://bitbucket.org/bohrium/bohrium/get/7750fe940c8bd5d0d8c1efe24f9a5faace7d6c91.zip"
  version "v0.2-1727-g7750fe9"
  sha1 "53b85c89c32e07641b5f7568f90b3c84a0a24c66"

  depends_on "cmake" => :build
  depends_on "mono" => :build
  depends_on "swig" => :build
  depends_on "Python" => :build
  depends_on "boost" => [:build,  "universal"]
  depends_on "cheetah" => [:python, "Cheetah.Template", :build]

  head do
    url "https://bitbucket.org/bohrium/bohrium.git"
  end

  def install
    if build.head?
      ln_s cached_download/".git", ".git"
    end

    # Set the python-path to also pick up the Brew-installed items, as pip will install there
    ENV["PYTHONPATH"] = ENV["PYTHONPATH"] + ":/usr/local/lib/python2.7/site-packages/"
    system "cmake", ".", *std_cmake_args
    system "make", "install"
    system "touch", "#{prefix}/var/bohrium/objects/.empty"
    system "touch", "#{prefix}/var/bohrium/kernels/.empty"
  end

  test do
    system "test/c/helloworld/bh_hello_world_c"
  end
end
