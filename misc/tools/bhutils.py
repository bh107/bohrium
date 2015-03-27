#!/usr/bin/env python
import ConfigParser
import json
import glob
import os
from os.path import join as pjoin, expanduser
import argparse
import shutil

def print_timings(times):
    _, start = times[0]
    for what, when in times:
        m, s = divmod(when-start, 60)
        h, m = divmod(m, 60)
        print "%s@%d:%02d:%02d" % (what, h, m, s)

def bytecode_format(bytecodes, indent):
    """
    Custom json-encoding for improved human-readability
    of the bytecode definition.
    """

    def nestedlist_format(outer):
        outer.sort()
        strings = []
        for inner in outer:
            strings.append(" "*indent*3+'[ "{0}" ]'.format('", "'.join(inner)))

        return ",\n".join(strings)

    opcode_fstr = """
{
    "opcode": "%s",
    "doc":  "%s",
    "code": "%s",
    "id":   "%s",
    "nop":   %d,
    "types": [
%s
    ],
    "layout": [
%s
    ],
    "elementwise":   %s,
    "system_opcode": %s
}"""

    bytecode_str = []
    for bytecode in bytecodes:
        bytecode_str.append(opcode_fstr % ( bytecode['opcode'], bytecode['doc'],
            bytecode['code'],
            bytecode['id'],
            bytecode['nop'],
            nestedlist_format(bytecode['types']),
            nestedlist_format(bytecode['layout']),
            str(bytecode['elementwise']).lower(),
            str(bytecode['system_opcode']).lower()
        ))

    return "[{0}\n]".format(','.join(bytecode_str))

def load_bytecode(path):
    """
    Load/Read the Bohrium bytecode definition from the Bohrium-sourcecode.

    Raises an exception if 'opcodes.json' and 'types.json' cannot be found or
    are invalid.

    Returns (opcodes, types)
    """
    if not path:
        path = os.sep.join(['..', '..'])

    opcodes = json.load(open(os.sep.join([
        path, 'core', 'codegen', 'opcodes.json'
    ])))
    types   = json.load(open(os.sep.join([
        path, 'core', 'codegen', 'types.json'
    ])))

    return (opcodes, types)

def load_config(path=None):
    """
    Load/Read the Bohrium config file and return it as a ConfigParser object.
    If no path is given the following paths are searched::

        /etc/bohrium/config.ini
        ${HOME}/.bohrium/config.ini
        ${CWD}/config.ini

    Raises an exception if config-file cannot be found or is invalid.

    Returns config as a ConfigParser object.
    """

    if path and not os.path.exists(path):   # Check the provided path
        raise e("Provided path to config-file [%s] does not exist" % path)

    if not path:                            # Try to search for it
        potential_path = os.sep.join(['etc','bohrium','config.ini'])
        if os.path.exists(potential_path):
            path = potential_path

        potential_path = os.sep.join([expanduser("~"), '.bohrium',
                                      'config.ini'])
        if os.path.exists(potential_path):
            path = potential_path

        potential_path = os.environ["BH_CONFIG"] if "BH_CONFIG" in os.environ else ""
        if os.path.exists(potential_path):
            path = potential_path

    if not path:                            # If none are found raise exception
        raise e("No config-file provided or found.")

    p = ConfigParser.SafeConfigParser()         # Try and parse it
    p.read(path)

    return p

def import_bohrium():
    """Import/Load Bohrium with source-dumping enabled."""

    os.environ['BH_VE_CPU_JIT_ENABLED']     = "1"
    os.environ['BH_VE_CPU_JIT_PRELOAD']     = "1"
    os.environ['BH_VE_CPU_JIT_OPTIMIZE']    = "0"
    os.environ['BH_VE_CPU_JIT_FUSION']      = "0"
    os.environ['BH_VE_CPU_JIT_DUMPSRC']     = "1"
    import warnings
    import bohrium as np
    from bohriumbridge import flush
    warnings.simplefilter('error')

    return (np, flush)

def cmd_clean(args):
    conf = load_config()

    #Clean CPU's JIT cache
    try:
        s = conf.get("cpu", "object_path")
        files = glob.glob(pjoin(s, "*.so"))
        print "rm %s"%pjoin(s, "*.so")
        for f in files:
            os.remove(f)
    except ConfigParser.NoOptionError:
        pass

    #Clean the cache of all fusers
    for sec in conf.sections():
        try:
            if conf.get(sec, "type") == "fuser":
                s = conf.get(sec, "cache_path")
                print "rm -R %s"%s
                shutil.rmtree(s)
        except ConfigParser.NoOptionError:
            pass
        except OSError:
            pass

    #Clean the nvidia cache
    try:
        shutil.rmtree(pjoin(expanduser("~"), ".nv"))
        print "rm -R ~/.nv"
    except OSError:
        pass

def cmd_info(args):
    print "The Bohrium Execution Stack:"
    conf = load_config()
    comp = conf.get("bridge", "children")
    try:
        while True:
            print comp
            comp = conf.get(comp, "children")
    except ConfigParser.NoOptionError:
        pass

def main():

    def parser_bohrium_src(parser, path):
        """Check that 'path' points to the Bohrium source dir"""
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            return os.path.abspath(path)
        else:
            parser.error("The path %s does not exist!"%path)

    parser = argparse.ArgumentParser(description='Set of utility functions for Bohrium.')
#    parser.add_argument(
#        'bohrium_src',
#        help='Path to the Bohrium source-code.',
#        type=lambda x: parser_bohrium_src(parser, x)
#    )

    subparsers = parser.add_subparsers()
    parser_clean = subparsers.add_parser('clean', help='Cleanup of cache files such as the fuse and JIT cache.')
    parser_clean.set_defaults(func=cmd_clean)

    parser_info = subparsers.add_parser('info', help='Print Info.')
    parser_info.set_defaults(func=cmd_info)

    #Parse the args and call the relevant cmd_*() function
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
