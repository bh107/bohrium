#!/usr/bin/env python
"""This script was used initially to generate the opcodes.json fil.
   NB: This script is not part of the cphVB installation script. 
       It is only here as documentation of the initial creations of all the opcodes in cphVB.
"""
import json
import sys
import os

mapped = {          # Datatype mapping from numpy to CPHVB
    '?': 'CPHVB_BOOL',

    'b': 'CPHVB_INT8',
    'B': 'CPHVB_UINT8',

    'h': 'CPHVB_INT16',
    'H': 'CPHVB_UINT16',

    'i': 'CPHVB_INT32',
    'I': 'CPHVB_UINT32',

    'l': 'CPHVB_INT32',
    'L': 'CPHVB_UINT32',

    'q': 'CPHVB_INT64',
    'Q': 'CPHVB_UINT64',

    'f': 'CPHVB_FLOAT32',
    'd': 'CPHVB_FLOAT64',
}

FX = {}


def main():

    functions   = json.load(open('../compute/codegen/functions.json'))                 # Read the function definitions.
    ignore      = json.load(open('../compute/codegen/functions.ignore.json'))          # Read list of functions to ignore.

    sys.path.append( os.getcwd()+ "/../../bridge/numpy/numpy/core/code_generators/" )
    from ufunc_docstrings import docdict
                                                                    # Generate the cpp code.
    for (opcode, nin, nout, signatures) in [f for f in functions if f[0] not in ignore]:

        fname   = opcode.replace('CPHVB_','').lower()
        opcount = nin+nout
        try:
            doc = docdict[fname].split("\n")[1].lstrip()
        except:
            if opcode == 'CPHVB_IDENTITY':
                doc = "The identity function that returns the input value converted to the output data type."
            else:
                doc = 'A description of %s'%fname
    
        N = {'opcode':opcode,'nop':nin+nout,'doc':doc,'system_opcode':False}
        in_types = []
        out_types= []
        T = {}
        for sin, sout in signatures:
            tin    = [mapped[t] for t in sin if t in mapped][::-1]
            tout   = [mapped[t] for t in sout if t in mapped][::-1]
            if len(tin) and len(tout):
                tin = tin[0]
                if tin in T:
                    if tout[0] not in T[tin]:
                        T[tin].extend(tout)
                else:
                    T[tin] = tout 

        N['types'] = T
        FX[N['opcode']] = N
        assert nin <= 3
        assert nout <= 1

    #Speciel opcodes
    FX['CPHVB_NONE'] = {'opcode':"CPHVB_NONE",'nop':0,'types':{},'system_opcode':False,'doc':"A opcode that should be ignored."}

    #System opcodes
    FX['CPHVB_USERFUNC'] = {'opcode':"CPHVB_USERFUNC",'nop':0,'types':{},'system_opcode':True,'doc':"System instruction that represents a user-defined function registered by the Bridge."}
    FX['CPHVB_SYNC'] = {'opcode':"CPHVB_SYNC",'nop':1,'types':{},'system_opcode':True,'doc':"System instruction that informs the child component to make data synchronized and available."}
    FX['CPHVB_DISCARD'] = {'opcode':"CPHVB_DISCARD",'nop':1,'types':{},'system_opcode':True,'doc':"System instruction that informs the child component to forget the array."}
    FX['CPHVB_DESTROY'] = {'opcode':"CPHVB_DESTROY",'nop':1,'types':{},'system_opcode':True,'doc':"System instruction that informs the child component to deallocate the array."}

    ret = json.dumps(FX, sort_keys=True, indent=4)

    print """/*
 * Copyright 2012 Mads R. B. Kristensen <madsbk@gmail.com>
 *
 * This file is part of cphVB <http://code.google.com/p/cphvb/>.
 *
 * cphVB is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cphVB is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with cphVB. If not, see <http://www.gnu.org/licenses/>.
 */
  
 /*
 * This is the definition of all opcodes in cphVB. It uses the JSON 
 * (JavaScript Object Notation) syntax <http://json.org>.
 *
 * It contains a dictionary of all opcodes in the cphVB project. 
 * The format is mostly self-explaining but the "types" key needs some 
 * explanation. It contains a dictionary of all input data types 
 * supported by the associated opcode and each supported input type 
 * has a list of suppored output data types. 
 * Note that multiple inputs to an instruction should always have 
 * identical data types.
 *
 * The "system_opcode" specifies whether a given opcode is a regular 
 * arithmetic instruction or a special system instruction.
 */

    """ + ret



if __name__ == "__main__":
    main()
