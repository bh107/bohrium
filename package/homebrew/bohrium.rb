require "formula"

class Bohrium < Formula
  homepage "http://bh107.org/"
  head "https://github.com/bh107/bohrium.git"
  url "https://codeload.github.com/bh107/bohrium/zip/177d8d7c4178ac58568c3d79da8096d235a43bf2"
  version "v0.2-1889-g177d8d7"
  sha1 "51c6242aade09bcdcc63d451a9ea9e0b2934b3bc"

  depends_on "cmake" => :build
  depends_on "mono" => :build
  depends_on "swig" => :build
  depends_on "Python" => :build
  depends_on "boost" => [:build,  "universal"]
  depends_on "numpy" => :build
  depends_on "cython" => [:python, :build]
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
