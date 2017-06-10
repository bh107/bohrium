#!/usr/bin/env python
import json
import os
from os.path import join, exists
import argparse

def main(args):

    dllimport = "        [DllImport(\"libbhc\", SetLastError = true, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Auto)]\n        public extern static %s;\n\n"

    prefix = os.path.abspath(os.path.dirname(__file__))

    # Let's read the opcode and type files
    with open(join(prefix,'..','..','core','codegen','opcodes.json')) as f:
        opcodes = json.loads(f.read())
    with open(join(prefix,'..','..','core','codegen','types.json')) as f:
        types   = json.loads(f.read())
        type_map = {}
        for t in types[:-1]:
            type_map[t['enum']] = {'cpp'     : t['cpp'],
                                   'bhc'     : t['bhc'],
                                   'name'    : t['union'],
                                   'bhc_ary' : "bhc_ndarray_%s_p"%t['union']}

    # Let's generate the header and implementation of all array operations
    head = ""
    for op in opcodes:
        if op['opcode'] in ["BH_REPEAT", "BH_RANDOM", "BH_NONE"]:#We handle random separately and ignore None
            continue

        # Generate functions that takes no operands
        if len(op['types']) == 0:
            decl = "void bhc_%s()"%(op['opcode'][3:].lower())
            head += dllimport%decl

        # Generate a function for each type signature
        for type_sig in op['types']:
            for layout in op['layout']:

                for i in range(len(layout)):#We need to replace 1D symbols with A
                    if layout[i].endswith("D"):
                        layout[i] = "A"

                decl = "void bhc_%s"%(op['opcode'][3:].lower())
                assert len(layout) == len(type_sig)
                for symbol, t in zip(layout,type_sig):
                    decl += "_%s%s"%(symbol, type_map[t]['name'])
                decl += "([Out] %s @out"%type_map[type_sig[0]]['bhc_ary']
                for i, (symbol, t) in enumerate(zip(layout[1:], type_sig[1:])):
                    decl += ", "
                    if symbol == "A":
                        decl += "[In] %s in%d"%(type_map[t]['bhc_ary'],i+1)
                    else:
                        decl += "[In] %s in%d"%(type_map[t]['bhc'],i+1)
                decl += ")"
                head += dllimport%decl
        head += "\n\n"

    #Let's handle random
    decl = "void bhc_random123_Auint64_Kuint64_Kuint64([Out] bhc_ndarray_uint64_p @out, [In] bhc_uint64 seed, [In] bhc_uint64 key)"
    head += dllimport%decl

    #We also need flush
    decl = "void bhc_flush()"
    head += dllimport%decl

    #Let's add header and footer
    head = """#region Autogen notice
/****************************************************************
 *         This file is autogenerated, do not modify!           *
 ****************************************************************/
#endregion
#region Copyright
/*
This file is part of Bohrium and copyright (c) 2013 the Bohrium
team <http://www.bh107.org>.

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the
GNU Lesser General Public License along with Bohrium.

If not, see <http://www.gnu.org/licenses/>.
*/
#endregion
using System;
using System.Linq;
using System.Text;
using System.Collections.Generic;
using System.Runtime.InteropServices;

using bhc_bool = System.Boolean;
using bhc_int8 = System.SByte;
using bhc_uint8 = System.Byte;
using bhc_int16 = System.Int16;
using bhc_uint16 = System.UInt16;
using bhc_int32 = System.Int32;
using bhc_uint32 = System.UInt32;
using bhc_int64 = System.Int64;
using bhc_uint64 = System.UInt64;
using bhc_float32 = System.Single;
using bhc_float64 = System.Double;
using bhc_complex64 = NumCIL.Complex64.DataType;
using bhc_complex128 = System.Numerics.Complex;

namespace NumCIL.Bohrium
{
    internal static partial class PInvoke
    {

%s

    }
}
"""%head

    #Finally, let's write the files
    with open(join(args.output,'bhc_array_operations.cs'), 'w') as f:
        f.write(head)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description = 'Generates the array operation source files for the Bohrium C bridge.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'output',
        help='Path to the output directory.'
    )
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    main(args)
