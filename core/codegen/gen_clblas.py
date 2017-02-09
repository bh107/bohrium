#!/usr/bin/env python
import json
from string import Template
import time

import argparse
from argparse_utils import *

"""
Generates the clBLAS ext methods
"""

def gen_level3(level3, header_tpl, body_tpl, func_tpl, footer_tpl):
    # Create substitute dictionary
    # These consist of a 'if_%' and 'endif_%' for each options, dims, and matrix
    substitutes = {}
    for m in level3["methods"]:
        for o in m["options"] + m["dims"] + m["matrix"]:
            substitutes["if_%s" % o], substitutes["endif_%s" % o] = "/*", "*/"

    # Create body for Level 3
    lvl3_body = ""
    for method in level3["methods"]:
        # Everything in the actual method should be present in the template
        subs = substitutes.copy()
        for m in method["options"] + method["matrix"] + method["dims"]:
            subs["if_%s" % m], subs["endif_%s" % m] = "", ""

        # Grab general options
        options = level3["options"].copy()

        # Some options might be overridden from the JSON-file
        if "overrides" in method:
            for t in method["overrides"].keys():
                options[t] = dict(options[t], **method["overrides"][t])

        # Create each inner function body
        lvl3_body_func = ""
        for t in method["types"]:
            lvl3_body_func += func_tpl.substitute(
                utype=options[t]["type"].upper(),
                t=t,
                **(dict(subs, **dict(options[t], **method)))
            )

        lvl3_body += body_tpl.substitute(
            uname=method["name"].capitalize(),
            second_matrix=method["matrix"][1],
            func=lvl3_body_func,
            **(dict(subs, **method))
        )

    # Create the footer
    lvl3_footer = ""
    for method in level3["methods"]:
        lvl3_footer += footer_tpl.substitute(
            uname=method["name"].capitalize(),
            **method
        )

    # Replace 'body' and 'footer' in the 'header' template
    return header_tpl.substitute(
        timestamp=time.strftime("%d/%m/%Y %H:%M"),
        body=lvl3_body,
        footer=lvl3_footer
    )

def main(args):
    data  = open(args.template_directory + "/methods.json").read()
    json_data = json.loads(data)

    header_level3    = Template(open(args.template_directory + "/header_level3.tpl").read())
    body_level3      = Template(open(args.template_directory + "/body_level3.tpl").read())
    body_func_level3 = Template(open(args.template_directory + "/body_func_level3.tpl").read())
    footer_level3    = Template(open(args.template_directory + "/footer_level3.tpl").read())

    level3 = gen_level3(json_data["level3"], header_level3, body_level3, body_func_level3, footer_level3)
    args.level3_cpp.write(level3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generates level3.cpp for clBLAS ext methods'
    )

    parser.add_argument(
        'template_directory',
        type=is_dir,
        action=FullPaths,
        help="The template directory, which contains the json file for autogenerating clBLAS methods."
    )

    parser.add_argument(
        'level3_cpp',
        type=argparse.FileType('w'),
        help="The level3.cpp to write."
    )

    main(parser.parse_args())
