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

out = ""#The output of this build and/or testing
#Execute command and capture output
def cmd(*args, **kwarg):
    global out
    if 'stderr' not in kwarg:
        kwarg['stderr'] = STDOUT#Default value
    ret = check_output(*args, **kwarg)
    out += ret
    return ret

def main(args):
    global out
    #Lets update the repos
    ret = cmd(['git','pull'], cwd=args.cmake_file_dir)
    if args.only_on_changes and 'Already up-to-date' in ret:
        out += "No changes to the git repos, exiting."
        return
    cmd(['git','checkout', args.branch], cwd=args.cmake_file_dir)

    #Make and change to a tmp dir
    tmpdir = tempfile.mkdtemp(prefix="bh_deb_builder_")
    os.chdir(tmpdir)

    cmd(['cmake', args.cmake_file_dir, '-DCPACK_PACKAGE_CONTACT=%s'%args.contact])

    #Lets find the change files generatored by cmake
    m = re.findall("signfile (.*source\.changes) %s\s*Successfully signed "\
                   "dsc and changes files"%args.contact, out)
    if len(m) <= 0:
        raise RuntimeError("cmake didn't generate any deb-src change files!")
    out += "\ncmake generated the following deb-src change files: %s\n"%str(m)

    #Lets uploade the files
    for changefile in m:
        out += "Uploading %s\n"%changefile
        cmd(['dput', 'bohrium-nightly', 'Debian/%s'%changefile])

    #Lets cleanup
    cmd(['rm','-Rf',tmpdir])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                description='Build the debian source packages and upload them to launchpad.net.',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--cmake-file-dir',
        default=str(path.dirname(path.abspath(__file__))),
        type=str,
        help='Path to directory where the CMakeLists.txt file is.'
    )
    parser.add_argument(
        '--branch',
        default="master",
        type=str,
        help='The git branch to pull before building.'
    )
    parser.add_argument(
        '--contact',
        default="Bohrium Builder <builder@bh107.org>",
        type=str,
        help='The package contact info which is also used for signing the package.'
    )
    parser.add_argument(
        '--email',
        type=str,
        help='The result of the build and/or test will be emailed to the specified address.'
    )
    parser.add_argument(
        '--only-on-changes',
        action="store_true",
        help='Only execute when the git repos has been changed.'
    )
    args = parser.parse_args()
    status = "SUCCESS"
    try:
        main(args)
    except StandardError, e:
        out += "*"*70
        out += "\nERROR: %s"%traceback.format_exc()
        out += "*"*70
        out += "\n"
        status = "FAILURE"
        try:
            out += e.output
        except:
            pass
    print out
    if args.email:
        print "send status email to '%s'"%args.email
        p = Popen(['mail','-s','"[Bohrium PPA Build] The result of build was a %s"'%status, args.email],
                  stdin=PIPE)
        p.communicate(input=out)

