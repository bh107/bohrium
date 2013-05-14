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
import string

from parse import *

def pprint_table(data, just=None, spacer=['-', '+', '*', '=', ' ', '_', '#']):
    """Print in justified columns."""

    # Default justification, first column is leftmost, remaining are right-most
    justify = [string.ljust] + [string.rjust]*(len(data)-1) if not just else just

    def format_row(row, width, just, spacer):
        """Format row according to justification and expand spacers to fill line."""

        for c, word in enumerate(row):
            if word in spacer:
                yield word * width[c] + (word if c+1 < len(width) else '')
            else:
                yield justify[c](str(word), width[c])+ (' ' if c+1 < len(width) else '')

                                        # Compute max-width of each column
    margin = 2                          # Plus margin
    widths      = ((c, len(str(word))+margin) for row in data for c, word in enumerate(row))
    col_widths  = [0]*len(data[0])
    for col, width in widths:
        col_widths[col] = width if width > col_widths[col] else col_widths[col]

    for row in data:                    # Actually print it
        print ''.join(word for word in format_row(row, col_widths, just,spacer))

def estimate_flop(instruction):
    """This seems like a reasonable estimate for element-wise operations.
    It does however not fit ufuncs..."""

    return reduce(lambda x,y: x*y, instruction.outputs()[0].shape)

def count(fn, exclude):
    """Counts opcodes and FLOPs."""

    p = Parser(fn)

    flops   = {}                    # Count flops
    opcodes = {}                    # Count opcodes
    for instruction in (i for i in p.parse() if i.opcode not in exclude):
        opcode = instruction.opcode
        flops[opcode]   = flops[opcode]+estimate_flop(instruction) if opcode in flops else estimate_flop(instruction)
        opcodes[opcode] = opcodes[opcode]+1 if opcode in opcodes else 1

    db = sorted(                    # List them as: (opcode, count, flops)
        ((opc, opcodes[opc], flops[opc]) for opc in opcodes.keys()),
        key=lambda x: x[2],         # Sort by flops
        reverse=True                # Ascending
    )

    return db

def group(entries):

    counts = {'mem': [],'ext':[],'elem':[]}
    aggr_i = {'mem': [],'ext':[],'elem':[]}
    aggr_f = {'mem': [],'ext':[],'elem':[]}

    for entry in entries:       # Group results
        opcode, _, _ = entry
        if opcode in ["FREE", "DISCARD", "SYNC"]:
            counts['mem'].append(entry)
        elif opcode in ["USERFUNC"]:
            counts['ext'].append(entry)
        else:
            counts['elem'].append(entry)

                                # Aggregate
    for instr_type in ['elem', 'mem', 'ext']:
        
        aggr_i[instr_type] = reduce(
            lambda carry, (x,y,z): carry+y,
            counts[instr_type],
            0
        )
        aggr_f[instr_type] = reduce(
            lambda carry, (x,y,z): carry+z,
            counts[instr_type],
            0
        )

    return (counts, aggr_i, aggr_f)

def main():
                                                    # Setup argument parser
    p = argparse.ArgumentParser(
        description='Counts instructions and FLOPs (FLoating-point OPerations).'
    )
    p.add_argument(
        'filename',
        help='Path / filename of the trace-file'
    )
    p.add_argument(
        '--exclude',
        nargs='?',
        default=[],
        help="List of opcodes to exclude from parsing.\nExample: --exclude FREE,DISCARD,SYNC"
    )
    args = p.parse_args()                           # Grab arguments
                                                    
    tracefile = args.filename                       # Check that tracefile exists
    if not os.path.exists(tracefile) or \
       not os.path.isfile(tracefile):
        return "Error: invalid filename <%s>." % tracefile

    stats = count(tracefile, args.exclude)          # Count and sort
    counts, aggr_i, aggr_f = group(stats)           # Group
    total_i = sum(aggr_i[key] for key in aggr_i)    # Aggr total
    total_f = sum(aggr_f[key] for key in aggr_f)

    table = []                                      # Build list to render
    table += [['Element-wise',' ',' ']]             # Element-wise
    table += [['-','-','-']]
    table += counts['elem']
    table += [['-','-','-']]
    table += [['Subtotal', aggr_i['elem'], aggr_f['elem']]]
    table += [[' ', '=', '=']]

    table += [['Extensions', ' ', ' ']]             # Extensions
    table += [['-','-','-']]
    table += counts['ext']
    table += [['-','-','-']]
    table += [['Subtotal', aggr_i['ext'], aggr_f['ext']]]
    table += [[' ', '=', '=']]

    table += [['System',' ',' ']]                   # System
    table += [['-','-','-']]
    table += counts['mem']
    table += [['-','-','-']]
    table += [['Subtotal', aggr_i['mem'], aggr_f['mem']]]
    table += [[' ', '=', '=']]

    table += [[' ', ' ', ' ']]                      # Total
    table += [['Total', total_i, total_f]]
    table += [[' ', '=', '=']]

    return table, ""
    
if __name__ == "__main__":
    out, err = main()
    if err:
        print "Error: %s" % err
    if out:
        pprint_table(out)

