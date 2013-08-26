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
import itertools
import pprint
import re
import os
import sys
import argparse
import subprocess

class Instruction(object):

    def __init__(self, opcode, operands, order):
        try:
            self.opcode = opcode
            self.o_ops  = [operands[0]]
            self.i_ops  = [] + operands[1:]
            self.order  = order

            if (opcode in ["SYNC", "FREE", "DISCARD"]):
                self.i_ops.append(operands[0])
        except:
            print "Something went unbelievably wrong! [%s,%s,%s]" % (str(opcode), str(operands), str(order))

    def __str__(self):
        op_dots = "\n".join([op.dot() for op in self.operands])
        return "%s %s" % (self.opcode, ', '.join((str(op) for op in self.operands)))

    def operands(self):
        return self.o_ops + self.i_ops

    def inputs(self):
        """Returns a list of operands."""
        return self.i_ops

    def outputs(self):
        """Returns a list of output-operands."""
        return self.o_ops

    def ref(self):
        return self.opcode

    def dot(self):
        return '[shape=box, style=filled, fillcolor="#CBD5E8", label="%s\\n%d"]' % (self.opcode, self.order)

class Operand(object):

    def __init__(self, symbol):
        self.symbol = symbol

    def ref(self):
        return self.symbol

    def b(self):
        return self.symbol

class Base(Operand):

    def __init__(self, symbol, addr, dims, start, shape, stride, dtype, data):

        self.addr   = addr
        self.dims   = int(dims)
        self.start  = int(start)
        self.shape  = [int(x) for x in shape.split(',')]
        self.stride = [int(x) for x in stride.split(',')]
        self.dtype  = dtype
        self.data   = data

        super(Base, self).__init__(symbol)

    def b(self):
        return self.addr

    def __str__(self):
        return 'Base-%s' % self.addr

    def dot(self):
        return '[shape=box, style="rounded,filled", fillcolor="#B3E2CD", label=%s]' % self.ref()

class View(Operand):

    def __init__(self, symbol, addr, dims, start, shape, stride, dtype, base):

        self.addr   = addr
        self.dims   = int(dims)
        self.start  = int(start)
        self.shape  = [int(x) for x in shape.split(',')]
        self.stride = [int(x) for x in stride.split(',')]
        self.dtype  = dtype
        self.base   = base

        super(View, self).__init__(symbol)

    def b(self):
        return self.base.addr

    def __str__(self):
        return 'View-%s' % self.ref()

    def dot(self):
        return '[shape=box, style="rounded, filled", fillcolor="#E6F5C9", label=%s]' % self.ref()

class Constant(Operand):

    def __str__(self):
        return "Const-%s" % (self.symbol)

    def dot(self):
        return '[shape=box, style="rounded, filled", fillcolor="#F4CAE4", label=%s]' % self.symbol

class Parser(object):

    re_instr    = "BH_(?P<OPCODE>\w+)\sOPS=(?P<N_OPS>\d+)"
    re_meta     = "\s+(?P<OPN>\w+)?\s+\[(?:(?:\s+Addr:\s+(?P<ADDR>\w+)\s+Dims:\s+(?P<DIMS>\d+)\s+Start:\s+(?P<START>\d+)\s+Shape:\s+(?P<SHAPE>[\d,]+)\s+Stride:\s+\s+(?P<STRIDE>[\d,]+)\s+Type:\s+(?P<TYPE>\w+)\s+Data:\s(?P<DATA>.*?),\s+Base:\s+(?P<BASE>.*?)\s+)|(?:\s+CONST=(?P<CONST>[\d.,\-~]+)\s+))"
    re_meta     = "\s+(?P<OPN>\w+)?\s\[(?:(?:\s+Dims:\s+(?P<DIMS>\d+)\sStart:\s+(?P<START>\d+)\s+Shape:\s+(?P<SHAPE>[\d,]+)\s+Stride:\s+\s+(?P<STRIDE>[\d,]+)\sBase=>\[\sAddr:\s(?P<ADDR>.*?)\s+Type:\s+(?P<TYPE>\w+)\s#elem:\s(?P<ELEMN>[\d]+)\sData:\s(?P<DATA>.*?)\s\])|(?:\s+CONST\((?P<CONST_TYPE>BH_.*)\)=(?P<CONST>[\d.,\-~]+)\s+)\])"

    def __init__(self, path):

        self._path  = path

        self._names   = {}
        self._symfst  = 97
        self._symlst  = 122
        self._symcur  = self._symfst
        self._symrnd  = 0

        self._consts = 0
        self._inc    = 0
        self._rank   = 0

        self._instructions = []

    def _symbolize(self, addr):

        if addr not in self._names:                   # Create a symbol for the address.
            if self._symcur > self._symlst:
                self._symcur = self._symfst
                self._symrnd += 1
            self._names[addr] = "%s%d" % (chr(self._symcur), self._symrnd) if self._symrnd > 0 else chr(self._symcur)
            self._symcur += 1

        return self._names[addr]

    def _desymbolize(self, operand):

        self._names.pop(operand.addr, None)

    def _tuplify(self, lines):
        """Generate a list of 'Instruction' objects; includes symbol translation."""

        instructions = []

        i = 0
        count = 0   # Instruction count, as to maintain the sequential order
        while( i<len(lines) ):

            m = re.match(Parser.re_instr, lines[i], re.MULTILINE + re.DOTALL)
            i += 1

            if m:                           # We got an instruction
                opcode, n_ops = (m.group('OPCODE'), int(m.group('N_OPS')))
                operands = []
                op_line = i
                while(True):                # Parse instruction operands

                    op_m = re.match(Parser.re_meta, lines[i], re.DOTALL)
                    if not op_m:            # No match -> no more operands
                        break

                    op_n, ndims, start, shape, stride, addr, dtype, elemn, data, c_type, c_value= op_m.groups()
                    i += 1

                    if c_value:                            # Constant
                        operands.append( Constant( "%0.2f" % float(c_value) ) )
                    elif "(nil)" in data or "0x0" in data:  # Base
                        operands.append( Base(self._symbolize(addr), addr, ndims, start, shape, stride, dtype, data) )
                    else:                                   # View
                        view = View(self._symbolize(addr), addr, ndims, start, shape, stride, dtype, data)

                        base_m = re.match(Parser.re_meta, lines[i], re.DOTALL)
                        op_n, ndims, start, shape, stride, addr, dtype, elemn, data, c_type, c_value= op_m.groups()
                        i += 2

                        base = Base(self._symbolize(addr), addr, ndims, start, shape, stride, dtype, data)
                        view.base = data

                        operands.append( view )

                if len(operands) < 1:
                    raise Exception("Failed parsing operands around line #%d." % op_line)

                instr = Instruction(opcode, operands, count)
                instructions.append( instr )
                count += 1

                # When a view is discarded the address no longer refers to the same symbol
                if 'DISCARD' in instr.opcode:
                    for op in instr.operands():
                        self._desymbolize( op )

        return instructions

    def parse(self):

        with open( self._path ) as fd:
            self._instructions = self._tuplify( fd.readlines() )

        return self._instructions

    def _edge(self, l, r, head="none"):
        """Do a 'dot-string' for connecting to entities."""
        return "%s -> %s [arrowhead=%s]\n" % (l, r, head)

    def dotify_list(self, instructions, exclude=[]):
        """Create a dot-representation of an instruction-list."""

        dots = "digraph G {\n"

        prevs    = []
        prevs_id = []

        # This code needs documentation...
        for instr in (i for i in instructions if i.opcode not in exclude):

            self._inc += 1
            instr_id    = "%s%d" % (instr.ref(), self._inc)
            instr_style = "%s %s\n" % (instr_id, instr.dot())

            styles  = instr_style
            graph   = ""

            self._rank += 1

            op_ids = []

            for op in instr.operands(): # Draw the operands

                i = self._inc
                self._inc += 1
                op_id       = "%s%d" % (op.ref(), self._inc )
                op_ids.append( (op.ref(), op_id) )

                op_style    = "%s %s\n" % (op_id, op.dot())

                styles  += op_style
                graph   += self._edge( instr_id, op_id )

                if 'base' in op.__dict__ and op.base:

                    base_id     = "%s%d" % (op.base.ref(), self._rank)
                    base_style  = "%s %s\n" % (base_id, op.base.dot())
                    styles      += base_style
                    graph       += self._edge( op_id, base_id )

            dots += "%s%s\n" % (styles, graph)
            op_ids = dict(op_ids)

            prev_ind = (len(prevs)-1)
            if prev_ind >= 0:

                prev    = prevs[prev_ind]
                prev_id = prevs_id[prev_ind]

                for (ps, cs) in itertools.product(prev.outputs(), instr.inputs()):
                    #if ps.symbol == cs.symbol:
                    if ps.b() == cs.b():
                        dots += self._edge( op_ids[cs.symbol], prev_id )

            prev    = instr
            prev_id = instr_id

            prevs.append( instr )
            prevs_id.append( instr_id )

        dots += "}"

        return dots

def dot_to_file(filename, dotstring, formats = ["svg", "fig", "xdot"]):
    """Call dot to convert dot-string into one a file."""

    cmd = None
    try:
        p = subprocess.Popen(["which", "dot"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        out.strip()
        dot = p.returncode
        cmd = out.strip()
    except:
        pass
    
    if cmd:
        errors = ''
        output = ''
        for f in formats:
            proc = subprocess.Popen([cmd, "-T", f, "-o", "%s.%s" % (filename, f)], stdin=subprocess.PIPE)
            out, err = proc.communicate(dotstring)
            output += out if out else ''
            errors += err if err else ''
        return (output, errors)
    else:
        return ("", "Could not find 'dot' on your machine.")

def main():

    p = argparse.ArgumentParser(description='Creates a .svg representation of a trace-file')
    p.add_argument(
        'filename',
        help='Path / filename of the trace-file'
    )
    p.add_argument(
        '--output',
        default="./",
        help="Where to dump the output."
    )
    p.add_argument(
        '--exclude',
        nargs='+',
        default=[],
        help="List of opcodes to exclude from parsing.\nExample: FREE,DISCARD,SYNC"
    )
    p.add_argument(
        '--formats',
        nargs='+',
        default=["svg"],
        help="List output formats for the visualized tree. See 'man dot' for supported formats"
    )

    args = p.parse_args()

    if not os.path.exists(args.filename) or not os.path.isfile(args.filename):
        return "Error: invalid filename <%s>." % args.filename

    if not os.path.exists(args.output) or os.path.isfile(args.output):
        return "Error: invalid output directory: <%s>" % args.output

    tracefile = args.filename
    output_fn = "%s%s%s" % (args.output, os.sep,
                            os.path.splitext(os.path.basename(tracefile))[0])

    p = Parser( args.filename )
    dotdata = p.dotify_list(p.parse(), args.exclude)
    return dot_to_file(output_fn, dotdata, args.formats)
    
if __name__ == "__main__":

    out = None
    err = None
    try:
        out, err = main()
    except Exception as e:
        err = str(e)

    if err:
        print "Error: %s" % err
    if out:
        print "Info: %s" % out

