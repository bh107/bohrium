#!/usr/bin/env python
import json
import os
from pprint import pprint
from Cheetah.Template import Template

def main():

    path    = "../../../core/codegen/types.json"
    types   = json.load(open(path))

    print bhtype_to_ctype(types)
    print ""
    print bhtype_to_shorthand(types)
    print ""
    print enumstr_to_ctypestr(types)
    print ""
    print enumstr_to_shorthand(types)


def bhtype_to_ctype(types):

    mapping = [(t["enum"], t["cpp"]) for t in types]
    template = Template(
        file        = "%s%s%s" % ("templates", os.sep, "bhtype_to_ctype.tpl"),
        searchList  = [{'types': mapping}]
    )
    return str(template)

def bhtype_to_shorthand(types):

    mapping = [(t["enum"], t["shorthand"]) for t in types]
    template = Template(
        file        = "%s%s%s" % ("templates", os.sep, "bhtype_to_shorthand.tpl"),
        searchList  = [{'types': mapping}]
    )
    return str(template)

def enumstr_to_ctypestr(types):

    mapping = [(t["enum"], t["c"]) for t in types]
    template = Template(
        file        = "%s%s%s" % ("templates", os.sep, "enumstr_to_ctypestr.tpl"),
        searchList  = [{'types': mapping}]
    )
    return str(template)

def enumstr_to_shorthand(types):

    mapping = [(t["enum"], t["c"]) for t in types]
    template = Template(
        file        = "%s%s%s" % ("templates", os.sep, "enumstr_to_shorthand.tpl"),
        searchList  = [{'types': mapping}]
    )
    return str(template)

if __name__ == "__main__":
    main()
