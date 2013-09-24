#!/usr/bin/env python
import json
import os
from pprint import pprint
from Cheetah.Template import Template

def main():

    types = [(t['enum'], t['shorthand'], t['ctype']) for t in json.load(open("bohrium_types.json"))]
    template = Template(
        file="%s%s%s" % ("templates", os.sep, "utils.tpl"),
        searchList=[{'types': types}]
    )
    print str(template)

if __name__ == "__main__":
    main()
