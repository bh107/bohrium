#!/usr/bin/env python
import ConfigParser
import subprocess
import argparse
import os

def read_config(path=None):
    """
    Read the Bohrium config file and return it as a ConfigParser object.
    If no path is given the following paths are searched::
        
        /etc/bohrium/config.ini
        ${HOME}/.bohrium/config.ini
        ${CWD}/config.ini

    Raises an exception if config-file cannot be found or is invalid.
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

if __name__ == "__main__":

    read_config()

    """
    p = argparse.ArgumentParser(description='Bohrium Tool')
    p.add_argument(
        'config',
        help='Path to Bohrium config-file.'
    )
    p.add_argument(
        'bohrium',
        help='Path to Bohrium source-code.'
    )
    args = p.parse_args()

    config  = get_config(args.config)
    opcodes = json.load(open(os.sep.join([
        args.bohrium, 'core', 'codegen', 'opcodes.json'
    ])))
    types   = json.load(open(os.sep.join([
        args.bohrium, 'core', 'codegen', 'types.json'
    ])))
    """

