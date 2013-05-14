#!/usr/bin/env python
import json
import os
from pprint import pprint
from Cheetah.Template import Template

def main():

    data    = "../../../core/codegen/opcodes.json"
    opcodes = json.load(open(data))
    
    data = {                    # Group the opcodes
        'system': [
            opcode for opcode in opcodes \
            if opcode['system_opcode'] \
            and 'USERFUNC' not in \
            opcode['opcode']
        ],
        'extensions': [
            opcode for opcode in opcodes \
            if 'USERFUNC' in opcode['opcode']
        ],
        'reductions': [
            opcode for opcode in opcodes \
            if 'REDUCE' in opcode['opcode']
        ],
        'binary': [
            opcode for opcode in opcodes \
            if opcode['nop']==3 \
            and opcode['elementwise']
        ],
        'unary': [
            opcode for opcode in opcodes \
            if opcode['nop']==2 \
            and opcode['elementwise']
        ]
    }

    template = Template(
        file="%s%s%s" % ("templates", os.sep, "dispatch.tpl"),
        searchList=[data]
    )
    print str(template)

if __name__ == "__main__":
    main()
