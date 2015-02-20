#!/usr/bin/env python
import subprocess
from subprocess import check_output, check_call, Popen, PIPE, STDOUT
from datetime import datetime
import os
from os import path
import argparse
import tempfile
import re
import traceback
import shutil

BREW_FORMULA_TEMPLAT = """\
require "formula"

class Bohrium < Formula
  homepage "http://bh107.org/"
  head "https://bitbucket.org/bohrium/bohrium.git"
  url "%s"
  version "%s"
  sha1 "%s"

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
"""

SRC = path.join(path.dirname(os.path.realpath(__file__)),"..","..")


def bash_cmd(cmd, cwd=None):
    print cmd
    p = subprocess.Popen(
        cmd,
        stdout  = subprocess.PIPE,
        stderr  = subprocess.PIPE,
        shell = True,
        cwd=cwd
    )
    out, err = p.communicate()
    print out,
    print err,
    return out

def main(args):
    global SRC

    # Grab the version name and the current commit tag
    bh_version = bash_cmd("git describe --tags --long --match *v[0-9]* ", cwd=SRC).strip()
    bh_commit = bash_cmd("git rev-parse HEAD", cwd=SRC).strip()

    zip_url = "https://bitbucket.org/bohrium/bohrium/get/%s.zip" % bh_commit

    # Get the source
    bash_cmd("curl ""%s"" -o ""%s/%s.zip"""%(zip_url, args.output, bh_version), cwd=SRC)

    # Get the sha1 sum of the zip
    zip_sha1 = bash_cmd("shasum ""%s/%s.zip""" % (args.output, bh_version), cwd=SRC).strip().split(' ')[0]

    bash_cmd("rm ""%s/%s.zip"""%(args.output, bh_version), cwd=SRC)

    # Make the script
    targetfolder = path.dirname(os.path.realpath(__file__))
    with open(path.join(targetfolder, "bohrium.rb"), "w+") as f:
        f.write(BREW_FORMULA_TEMPLAT % (zip_url, bh_version, zip_sha1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                description='Build the debian source packages.',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--contact',
        default="Bohrium Builder <builder@bh107.org>",
        type=str,
        help='The package contact info which is also used for signing the package.'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='The output directory.'
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = tempfile.mkdtemp()

    print "output dir: ", args.output
    main(args)