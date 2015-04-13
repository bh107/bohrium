require "formula"

class Bohrium < Formula
  homepage "http://bh107.org/"
  head "https://github.com/bh107/bohrium.git"
  url "https://codeload.github.com/bh107/bohrium/zip/17223c43d58bd61225594ea420e4a28433de5fe4"
  version "v0.2-1891-g17223c4"
  sha1 "0f326016df589dbd5607df042d31884faf9561d1"

  depends_on "cmake" => :build
  depends_on "mono" => :build
  depends_on "swig" => :build
  depends_on "Python" => :build
  depends_on "boost" => [:build,  "universal"]
  depends_on "numpy" => :build
  depends_on "cython" => [:python, "cython", :build]
  depends_on "cheetah" => [:python, "Cheetah.Template", :build]

  head do
    url "https://github.com/bh107/bohrium.git"
  end

  def install
    if build.head?
      ln_s cached_download/".git", ".git"
    end

    # Set the python-path to also pick up the Brew-installed items, as pip will install there
    if ENV["PYTHONPATH"] == nil
      ENV["PYTHONPATH"] = "/usr/local/lib/python2.7/site-packages/"
    else
      ENV["PYTHONPATH"] = ENV["PYTHONPATH"] + ":/usr/local/lib/python2.7/site-packages/"
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
