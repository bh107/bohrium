#!/usr/bin/env python
#
# This file is part of Bohrium and copyright (c) 2012 the Bohrium team:
# http://cphvb.bitbucket.org
#
# Bohrium is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as 
# published by the Free Software Foundation, either version 3 
# of the License, or (at your option) any later version.
#
# Bohrium is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the 
# GNU Lesser General Public License along with Bohrium. 
#
# If not, see <http://www.gnu.org/licenses/>.
#
#!/usr/bin/env python
import argparse
import pprint
import string
import json
import os

def main():

    script_dir  = "."+os.sep
    bytecodes = json.loads(open(script_dir+'../../core/codegen/opcodes.json').read())
    
    types = {
        'unary':     0,
        'binary':    0,
        'reduction': 0,
        'extension': 0,
        'system'   : 0,
    }

    nops = {0: 0, 1: 0, 2: 0, 3: 0, 4:0}
    sigs = 0
    
    errors = []

    unaries  = []
    binaries = []

    for bytecode in (bytecode for bytecode in bytecodes if bytecode['opcode'] != 'BH_NONE'):
        pprint.pprint(bytecode)
        nops[bytecode['nop']] += 1;
        if bytecode['elementwise']:
            if bytecode['nop'] == 2:
                types['unary'] += 1
                unaries.append((bytecode['opcode'], bytecode['doc']))
            elif bytecode['nop'] == 3:
                binaries.append((bytecode['opcode'], bytecode['doc']))
                types['binary'] += 1
                sigs += len(bytecode['types'])
                sigs += len(bytecode['types'])
            else:
                errors.append(bytecode)

            sigs += len(bytecode['types'])
        elif bytecode['system_opcode']:
            if bytecode['opcode'] == "BH_USERFUNC":
                types['extension'] += 1
            else:
                types['system'] += 1
            sigs += 1
        elif not bytecode['system_opcode'] and not bytecode['elementwise']:
            types['reduction'] += 1
            sigs += len(bytecode['types'])
        else:
            types['system'] += 1
            sigs += 1

    types_total = 0
    for t in types:
        types_total += types[t]

    #print sigs, types, types_total, errors
    #pprint.pprint(unaries)

    for b

if __name__ == "__main__":
    main()
