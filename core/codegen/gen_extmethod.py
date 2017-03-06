#!/usr/bin/env python
import json
import time

import argparse
from argparse_utils import *

import sys
sys.path.append(os.path.abspath('../../../thirdparty'))
from pyratemp import Template

def gen_extmethod(json, header_tpl, body_tpl, func_tpl, footer_tpl):
    # Create substitute dictionary
    # These consist of a 'if_%' and 'endif_%' for each option
    substitutes = {}
    for m in json["methods"]:
        for o in m["options"]:
            substitutes["if_%s" % o] = False

    # Create body
    body = ""
    for method in json["methods"]:
        # Everything in the actual method should be present in the template
        subs = substitutes.copy()
        for m in method["options"]:
            subs["if_%s" % m] = True

        # Grab general options
        options = json["options"].copy()

        # Some options might be overridden from the JSON-file
        if "overrides" in method:
            for t in method["overrides"].keys():
                options[t] = dict(options[t], **method["overrides"][t])

        # Create each inner function body
        body_func = ""
        for t in method["types"]:
            body_func += func_tpl(
                utype=options[t]["type"].upper(),
                t=t,
                **(dict(subs, **dict(options[t], **method)))
            )

        body += body_tpl(
            uname=method["name"].capitalize(),
            func=body_func,
            **(dict(subs, **method))
        )

    # Create the footer
    footer = ""
    for method in json["methods"]:
        footer += footer_tpl(
            uname=method["name"].capitalize(),
            **method
        )

    # Replace 'body' and 'footer' in the 'header' template
    return header_tpl(
        timestamp=time.strftime("%d/%m/%Y %H:%M"),
        body=body,
        footer=footer
    )

def main(args):
    data      = open(args.template_directory + "/methods.json").read()
    json_data = json.loads(data)

    header    = Template(open(args.template_directory + "/header.tpl").read(), escape=None)
    body      = Template(open(args.template_directory + "/body.tpl").read(), escape=None)
    body_func = Template(open(args.template_directory + "/body_func.tpl").read(), escape=None)
    footer    = Template(open(args.template_directory + "/footer.tpl").read(), escape=None)

    source = gen_extmethod(json_data, header, body, body_func, footer)
    args.cpp.write(source)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generates ext method'
    )

    parser.add_argument(
        'template_directory',
        type=is_dir,
        action=FullPaths,
        help="The template directory, which contains the json file for autogenerating the methods."
    )

    parser.add_argument(
        'cpp',
        type=argparse.FileType('w'),
        help="The cpp-file to write."
    )

    main(parser.parse_args())
