require "formula"

class Bohrium < Formula
  homepage "http://bh107.org/"
  head "https://bitbucket.org/bohrium/bohrium.git"
  url "https://bitbucket.org/bohrium/bohrium/get/f695301b133e809c728b5dff5c3c6cb64ca08d03.zip"
  version "v0.2-1854-gf695301"
  sha1 "6dfdd6bf6f6abab400456c05dd7ae0e7a0f42d14"

  depends_on "cmake" => :build
  depends_on "mono" => :build
  depends_on "swig" => :build
  depends_on "Python" => :build
  depends_on "boost" => [:build,  "universal"]
  depends_on "numpy" => :build
  depends_on "cython" => [:python, :build]
  depends_on "cheetah" => [:python, "Cheetah.Template", :build]

  head do
    url "https://bitbucket.org/bohrium/bohrium.git"
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
