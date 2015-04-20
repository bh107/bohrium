require "formula"

class Bohrium < Formula
  homepage "http://bh107.org/"
  head "https://github.com/bh107/bohrium.git"
  url "https://codeload.github.com/bh107/bohrium/zip/7615b8804d5be119a1acf8d940d1cbb2e9444296"
  version "v0.2-1894-g7615b88"
  sha1 "5881ba585674b165132d3cb1a5d8ba54179f8a0c"

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
