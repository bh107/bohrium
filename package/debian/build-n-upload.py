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

log = ""#The output of this build and/or testing
def bash_cmd(cmd, cwd=None):
    global log
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
    log += cmd
    log += out
    log += err
    return out

def main(args):
    global log
    script_dir = path.dirname(args.build_script)
    #Lets update the repos
    ret = bash_cmd('git pull', cwd=script_dir)
    if args.only_on_changes and 'Already up-to-date' in ret:
        log += "No changes to the git repos, exiting."
        return
    bash_cmd('git checkout %s'%args.branch, cwd=script_dir)

    #Make and change to a tmp dir
    tmpdir = tempfile.mkdtemp(prefix="bh_deb_builder_")

    res = bash_cmd("python %s --output=%s"%(args.build_script, tmpdir))

    #Lets find the change files generatored by cmake
    m = re.findall("signfile (.*source\.changes) %s\s*Successfully signed "\
                   "dsc and changes files"%args.contact, res)
    if len(m) <= 0:
        raise RuntimeError("the build script didn't generate any deb-src change files!")
    log += "\nthe build script generated the following deb-src change files: %s\n"%str(m)

    #Lets uploade the files
    for changefile in m:
        log += "Uploading %s\n"%changefile
        bash_cmd('dput bohrium-nightly %s'%changefile, cwd=tmpdir)

    #Lets cleanup
    bash_cmd('rm -Rf %s'%tmpdir)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                description='Build the debian source packages and upload them to launchpad.net.',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--build-script',
        default=str(path.join(path.dirname(path.abspath(__file__)), "build-package.py")),
        type=str,
        help='Path to the build script.'
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
        log += "*"*70
        log += "\nERROR: %s"%traceback.format_exc()
        log += "*"*70
        log += "\n"
        status = "FAILURE"
        try:
            log += e.output
        except:
            pass
    print
    print log
    if args.email:
        print "send status email to '%s'"%args.email
        p = Popen(['mail','-s','"[Bohrium PPA Build] The result of build was a %s"'%status, args.email],
                  stdin=PIPE)
        p.communicate(input=log)

