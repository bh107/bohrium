#!/usr/bin/env python
from subprocess import Popen, PIPE, STDOUT
from os import path
import argparse
import traceback
import tempfile

log = ""  # The output of this build and/or testing


def script_path():
    """Returns the path to the dir this script is in"""
    return path.dirname(path.realpath(__file__))


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
    bh_dir = path.abspath(args.bohrium_root)
    boost_dir = path.abspath(args.boost_root)
    build_dir = tempfile.mkdtemp(prefix="bh_build_")
    install_dir = tempfile.mkdtemp(prefix="bh_install_")
    print ("bh_dir: %s" % bh_dir)
    print ("boost_dir: %s" % boost_dir)
    print ("build_dir: %s" % build_dir)
    print ("install_dir: %s" % install_dir)

    bash_cmd('cmake %s -DCMAKE_BUILD_TYPE=Debug -DCORE_LINK_FLAGS="-static-libgcc -static-libstdc++" '
             '-DBoost_NO_SYSTEM_PATHS=ON -DBoost_USE_STATIC_LIBS=ON -DBOOST_ROOT=%s -DCMAKE_INSTALL_PREFIX=%s '
             '-DFORCE_CONFIG_PATH=%s -DVE_OPENMP_COMPILER_OPENMP_SIMD=OFF -DCYTHON_OPENMP=OFF' %
             (bh_dir, boost_dir, install_dir, install_dir), cwd=build_dir)
    bash_cmd("make install", cwd=build_dir)
    bash_cmd('python %s/create_wheel.py --npbackend-dir %s/lib/python2.7/site-packages/bohrium/ '
             '--bh-install-prefix %s --config %s/config.ini bdist_wheel' % (script_path(), install_dir,
                                                                            install_dir, install_dir), cwd=build_dir)

    bash_cmd('python3 %s/create_wheel.py --npbackend-dir %s/lib/python3.5/site-packages/bohrium/ '
             '--bh-install-prefix %s --config %s/config.ini bdist_wheel' % (script_path(), install_dir,
                                                                            install_dir, install_dir), cwd=build_dir)

    if not args.no_upload:
        # Upload the package to <https://pypi.python.org/pypi>
        try:
            bash_cmd('twine upload --sign --identity "%s" dist/*' % args.gpg_identity, cwd=build_dir)
        except KeyboardInterrupt:
            print("KeyboardInterrupt")

    print ("Created wheel packages can be found here: '%s'" % path.join(build_dir, "dist"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                description='Build the conda package and upload them to anaconda.org/bohrium.',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--bohrium-root',
        default=None,
        type=str,
        help='Path to the root of Bohrium.'
    )
    parser.add_argument(
        '--boost-root',
        default=None,
        type=str,
        help='Path to the root of a static boost installation compiled with -fPIC.'
    )
    parser.add_argument(
        '--gpg-identity',
        default="Bohrium Builder <builder@bh107.org>",
        type=str,
        help='The PGP sign identity (specified in ~/.gnupg/pubring.gpg).'
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
    parser.add_argument(
        '--no-upload',
        action="store_true",
        help='Do not upload the wheel package.'
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

    if args.email:
        print ("send status email to '%s'" % args.email)
        p = Popen(['mail','-s','"[Bohrium PIP Build] The result of build was a %s"' % status, args.email],
                  stdin=PIPE)
        p.communicate(input=log)

