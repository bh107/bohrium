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
    out = ""
    try:
        p = Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=True, cwd=cwd)
        while p.poll() is None:
            t = p.stdout.readline()
            out += t
            print t,
        t = p.stdout.read()
        out += t
        print t,
        p.wait()
    except KeyboardInterrupt:
        p.kill()
        raise
    log = "%s%s%s" % (log, cmd, out)
    return out


def main(args):
    global log
    bh_dir = args.bohrium_root
    recipe = path.join(bh_dir, "package", "conda", "bohrium")
    print "bh_dir: %s" %bh_dir
    print "recipe: %s" %recipe

    # Update the repos
    ret = bash_cmd('git pull', cwd=bh_dir)
    if args.only_on_changes and 'Already up-to-date' in ret:
        log += "No changes to the git repos, exiting."
        return
    bash_cmd('git checkout %s'%args.branch, cwd=bh_dir)

    # Build the conda package
    ret = bash_cmd('conda build --croot /tmp/conda_build_tmp %s'%recipe)
    res = re.search("anaconda upload (.*)", ret)
    if res is None:
         print "anaconda upload not found in output:"
         print ret
         raise ValueError("anaconda upload not found in output")
    tarball = res.group(1)
    print "tarball: '%s'" % tarball

    if args.auth_token is not None:
        bash_cmd('anaconda -t %s upload -u bohrium %s'%(args.auth_token, tarball))
    else:
        print "No anaconda access token specified thus no upload"


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                description='Build the conda package and upload them to anaconda.org/bohrium.',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--bohrium-root',
        default=str(path.join(path.dirname(path.abspath(__file__)),"..","..")),
        type=str,
        help='Path to the root of Bohrium.'
    )
    parser.add_argument(
        '--auth-token',
        default=None,
        type=str,
        help='Anaconda access token.'
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

