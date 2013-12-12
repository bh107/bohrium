#!/usr/bin/env python
import ConfigParser
import json
import os

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

        potential_path = os.sep.join([os.path.expanduser("~"), '.bohrium',
                                      'config.ini'])
        if os.path.exists(potential_path):
            path = potential_path

        potential_path = os.environ["BH_CONFIG"] if "BH_CONFIG" in os.environ else ""
        if os.path.exists(potential_path):
            path = potential_path

    if not path:                            # If none are found raise exception
        raise e("No config-file provided or found.")

    p = ConfigParser.ConfigParser()         # Try and parse it
    p.read(path)

    return p

def import_bohrium():
    """Import/Load Bohrium with source-dumping enabled."""
    
    os.environ['BH_VE_CPU_JIT_ENABLED']     = "1"
    os.environ['BH_VE_CPU_JIT_PRELOAD']     = "1"
    os.environ['BH_VE_CPU_JIT_OPTIMIZE']    = "0"
    os.environ['BH_VE_CPU_JIT_FUSION']      = "0"
    os.environ['BH_VE_CPU_JIT_DUMPSRC']     = "1"
    import bohrium as np

    return np

