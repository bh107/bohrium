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
        self.opcode = opcode
        self.o_ops  = [operands[0]]
        self.i_ops  = [] + operands[1:]
        self.order  = order

        if (opcode in ["SYNC", "FREE", "DISCARD"]):
            self.i_ops.append(operands[0])

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
        self.dims   = dims
        self.start  = start
        self.shape  = shape
        self.stride = stride
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
        self.dims   = dims
        self.start  = start
        self.shape  = shape
        self.stride = stride
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
    re_meta     = "\s+(?P<OPN>\w+)?\s+\[(?:(?:\s+Addr:\s+(?P<ADDR>\w+)\s+Dims:\s+(?P<DIMS>\d+)\s+Start:\s+(?P<START>\d+)\s+Shape:\s+(?P<SHAPE>[\d,]+)\s+Stride:\s+\s+(?P<STRIDE>[\d,]+)\s+Type:\s+(?P<TYPE>\w+)\s+Data:\s(?P<DATA>.*?)\s+Base:\s+(?P<BASE>.*?)\s+)|(?:\s+CONST=(?P<CONST>[\d.,\-~]+)\s+))"

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
                while(True):                # Parse instruction operands

                    op_m = re.match(Parser.re_meta, lines[i], re.DOTALL)
                    if not op_m:            # No match -> no more operands
                        break

                    op_n, addr, ndims, start, shape, stride, dtype, data, base, constant = op_m.groups()
                    i += 1

                    if constant:                            # Constant
                        operands.append( Constant( "%0.2f" % float(constant) ) )
                    elif "(nil)" in base or "0x0" == base:  # Base
                        operands.append( Base(self._symbolize(addr), addr, ndims, start, shape, stride, dtype, data) )
                    else:                                   # View
                        view = View(self._symbolize(addr), addr, ndims, start, shape, stride, dtype, base)

                        base_m = re.match(Parser.re_meta, lines[i], re.DOTALL)
                        op_n, addr, ndims, start, shape, stride, dtype, data, base, constant = base_m.groups()
                        i += 2

                        base = Base(self._symbolize(addr), addr, ndims, start, shape, stride, dtype, data)
                        view.base = base

                        operands.append( view )

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

def dot_to_svg(filename, dotstring):
    """Call dot to convert dot-string into svg."""

    dot = None
    try:
        dot = subprocess.check_call(["which", "dot"], stdout=subprocess.PIPE)
    except:
        dot = None
        
    if dot == 0:
        proc = subprocess.Popen(["dot", "-T", "svg", "-o" + filename + ".svg"], stdin=subprocess.PIPE)
        out, err = proc.communicate(dotstring)
        return "%s,%s" %(out, err)
    else:
        return "Could not find 'dot' on your machine."

def main():

    p = argparse.ArgumentParser(description='Creates a .svg representation of a trace-file')
    p.add_argument(
        'filename',
        help='Path / filename of the trace-file'
    )
    p.add_argument(
        '--exclude',
        nargs='+',
        default=[],
        help="List of opcodes to exclude from parsing.\nExample: FREE,DISCARD,SYNC"
    )
    args = p.parse_args()

    if not os.path.exists(args.filename) or not os.path.isfile(args.filename):
        return "Error: invalid filename <%s>." % args.filename

    p = Parser( args.filename )
    dotdata = p.dotify_list(p.parse(), args.exclude)
    return dot_to_svg(args.filename, dotdata)
    
if __name__ == "__main__":
    print main()

