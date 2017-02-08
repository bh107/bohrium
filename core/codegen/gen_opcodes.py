#!/usr/bin/env python
import json
import time
import argparse

"""
    Generates the include/bh_opcode.h and core/bh_opcode
    based on the definitnion in /core/codegen/opcodes.json.
"""

def gen_headerfile( opcodes ):

    enums = ("        %s = %s,\t\t// %s" % (opcode['opcode'], opcode['id'], opcode['doc']) for opcode in opcodes)
    stamp = time.strftime("%d/%m/%Y")

    l = [int(o['id']) for o in opcodes]
    l = [x for x in l if l.count(x) > 1]
    if len(l) > 0:
        raise ValueError("opcodes.json contains id duplicates: %s"%str(l))

    l = [o['opcode'] for o in opcodes]
    l = [x for x in l if l.count(x) > 1]
    if len(l) > 0:
        raise ValueError("opcodes.json contains opcode duplicates: %s"%str(l))

    max_ops = max([int(o['id']) for o in opcodes])
    return """
/*
 * Do not edit this file. It has been auto generated by
 * ../core/codegen/gen_opcodes.py at __TIMESTAMP__.
 */

#ifndef __BH_OPCODE_H
#define __BH_OPCODE_H

#include "bh_type.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Codes for known oparations */
enum /* bh_opcode */
{
__OPCODES__

    BH_NO_OPCODES = __NO_OPCODES__, // The amount of opcodes
    BH_MAX_OPCODE_ID = __MAX_OP__   // The extension method offset
};

/* Number of operands for operation
 *
 * @opcode Opcode for operation
 * @return Number of operands
 */
int bh_noperands(bh_opcode opcode);

/* Text string for operation
 *
 * @opcode Opcode for operation
 * @return Text string.
 */
DLLEXPORT const char* bh_opcode_text(bh_opcode opcode);

/* Determines if the operation is a system operation
 *
 * @opcode The operation opcode
 * @return The boolean answer
 */
DLLEXPORT bool bh_opcode_is_system(bh_opcode opcode);

/* Determines if the operation is a reduction operation
 *
 * @opcode The operation opcode
 * @return The boolean answer
 */
DLLEXPORT bool bh_opcode_is_reduction(bh_opcode opcode);

/* Determines if the operation is an accumulate operation
 *
 * @opcode The operation opcode
 * @return The boolean answer
 */
DLLEXPORT bool bh_opcode_is_accumulate(bh_opcode opcode);

/* Determines if the operation is performed elementwise
 *
 * @opcode Opcode for operation
 * @return TRUE if the operation is performed elementwise, FALSE otherwise
 */
DLLEXPORT bool bh_opcode_is_elementwise(bh_opcode opcode);

/* Determines whether the opcode is a sweep opcode
 * i.e. either a reduction or an accumulate
 *
 * @opcode
 * @return The boolean answer
 */
DLLEXPORT bool bh_opcode_is_sweep(bh_opcode opcode);

#ifdef __cplusplus
}
#endif

#endif
""".replace('__TIMESTAMP__', stamp).replace('__OPCODES__', '\n'.join(enums)).replace('__NO_OPCODES__', str(len(opcodes))).replace('__MAX_OP__',str(max_ops))

def gen_cfile(opcodes):

    text    = ['        case %s: return "%s";' % (opcode['opcode'], opcode['opcode']) for opcode in opcodes]
    nops    = ['        case %s: return %s;' % (opcode['opcode'], opcode['nop']) for opcode in opcodes]
    sys_op    = ['        case %s: '%opcode['opcode'] for opcode in opcodes if opcode['system_opcode']]
    elem_op   = ['        case %s: '%opcode['opcode'] for opcode in opcodes if opcode['elementwise']]
    reduce_op = ['        case %s: '%opcode['opcode'] for opcode in opcodes if opcode['reduction']]
    accum_op  = ['        case %s: '%opcode['opcode'] for opcode in opcodes if opcode['accumulate']]
    stamp   = time.strftime("%d/%m/%Y")

    return """
/*
 * Do not edit this file. It has been auto generated by
 * ../core/codegen/gen_opcodes.py at __TIMESTAMP__.
 */

#include <stdlib.h>
#include <stdio.h>
#include <bh_opcode.h>
#include <bh_instruction.hpp>
#include <stdbool.h>

/* Number of operands for operation
 *
 * @opcode Opcode for operation
 * @return Number of operands
 */
int bh_noperands(bh_opcode opcode)
{
    switch(opcode)
    {
__NOPS__

    default:
        return 3;//Extension methods have 3 operands always
    }
}

/* Text descriptions for a given operation */
const char* _opcode_text[BH_NONE+1];
bool _opcode_text_initialized = false;

/* Text string for operation
 *
 * @opcode Opcode for operation
 * @return Text string.
 */
const char* bh_opcode_text(bh_opcode opcode)
{
    switch(opcode)
    {
__TEXT__

        default: return "Unknown opcode";
    }
}

/* Determines if the operation is a system operation
 *
 * @opcode The operation opcode
 * @return The boolean answer
 */
bool bh_opcode_is_system(bh_opcode opcode)
{
    switch(opcode)
    {
__SYS_OP__
            return true;

        default:
            return false;
    }
}

/* Determines if the operation is an elementwise operation
 *
 * @opcode The operation opcode
 * @return The boolean answer
 */
bool bh_opcode_is_elementwise(bh_opcode opcode)
{
    switch(opcode)
    {
__ELEM_OP__
            return true;

        default:
            return false;
    }
}

/* Determines if the operation is a reduction operation
 *
 * @opcode The operation opcode
 * @return The boolean answer
 */
bool bh_opcode_is_reduction(bh_opcode opcode)
{
    switch(opcode)
    {
__REDUCE_OP__
            return true;

        default:
            return false;
    }
}

/* Determines if the operation is an accumulate operation
 *
 * @opcode The operation opcode
 * @return The boolean answer
 */
bool bh_opcode_is_accumulate(bh_opcode opcode)
{
    switch(opcode)
    {
__ACCUM_OP__
            return true;

        default:
            return false;
    }
}

/* Determines whether the opcode is a sweep opcode
 * i.e. either a reduction or an accumulate
 *
 * @opcode
 * @return The boolean answer
 */
bool bh_opcode_is_sweep(bh_opcode opcode)
{
    return (bh_opcode_is_reduction(opcode) || bh_opcode_is_accumulate(opcode));
}

""".replace('__TIMESTAMP__', stamp)\
   .replace('__NOPS__', '\n'.join(nops))\
   .replace('__TEXT__', '\n'.join(text))\
   .replace('__SYS_OP__', '\n'.join(sys_op))\
   .replace('__ELEM_OP__', '\n'.join(elem_op))\
   .replace('__REDUCE_OP__', '\n'.join(reduce_op))\
   .replace('__ACCUM_OP__', '\n'.join(accum_op))

def main(args):

    # Read the opcode definitions from opcodes.json.
    opcodes = json.loads(args.opcode_json.read())

    # Write the header file
    hfile = gen_headerfile(opcodes)
    args.opcode_h.write(hfile)

    # Write the c file
    cfile = gen_cfile(opcodes)
    args.opcode_cpp.write(cfile)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generates bh_opcode.cpp and bh_opcode.h')
    parser.add_argument(
        'opcode_json',
        type=argparse.FileType('r'),
        help="The opcode.json file that defines all Bohrium opcodes."
    )
    parser.add_argument(
        'opcode_h',
        type=argparse.FileType('w'),
        help="The bh_opcode.h to write."
    )
    parser.add_argument(
        'opcode_cpp',
        type=argparse.FileType('w'),
        help="The bh_opcode.cpp to write."
    )
    main(parser.parse_args())
