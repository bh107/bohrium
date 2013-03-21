#!/usr/bin/env python
import itertools
import pprint
import re
import os
import sys
import subprocess

class Instruction(object):

    def __init__(self, opcode, operands):

        self.opcode     = opcode
        self.operands   = operands

    def __str__(self):

        op_dots = "\n".join([op.dot() for op in self.operands])
        return "%s %s" % (self.opcode, ', '.join((str(op) for op in self.operands)))

    def ops(self):
        """Returns a list of operands."""
        return self.operands()

    #def out(self):
    #    """Returns a list of output-operands."""
    #    return self.operands()
    
    #def in(self):
    #   """Returns a list of input-operands."""
    #    return self.operands()

    def ref(self):
        return self.opcode

    def dot(self):

        return '[shape=box, style=filled, fillcolor="#CBD5E8", label=%s]' % self.opcode

class Operand(object):

    def __init__(self, symbol):
        self.symbol = symbol

    def ref(self):
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

        instructions = []

        i = 0
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

                    if constant:            # Constant
                        operands.append( Constant( "%0.2f" % float(constant) ) )
                    elif "(nil)" in base or "0x0" == base:   # Base
                        operands.append( Base(self._symbolize(addr), addr, ndims, start, shape, stride, dtype, data) )
                    else:                   # View
                        view = View(self._symbolize(addr), addr, ndims, start, shape, stride, dtype, base)

                        base_m = re.match(Parser.re_meta, lines[i], re.DOTALL)
                        op_n, addr, ndims, start, shape, stride, dtype, data, base, constant = base_m.groups()
                        i += 2

                        base = Base(self._symbolize(addr), addr, ndims, start, shape, stride, dtype, data)
                        view.base = base

                        operands.append( view )

                instr = Instruction(opcode, operands)
                instructions.append( instr )

                # When a view is discarded the address no longer refers to the same symbol
                if 'DISCARD' in instr.opcode:
                    for op in instr.operands:
                        self._desymbolize( op )

        return instructions

    def parse(self):

        with open( self._path ) as fd:
            self._instructions = self._tuplify( fd.readlines() )

        return self._instructions

    def _edge(self, l, r, head="none"):
        return "%s -> %s [arrowhead=%s]\n" % (l, r, head)

    def dotify_list(self, instructions):
        """Create a dot-representation of an instruction-list."""

        dots = "digraph G {\n"

        prevs    = []
        prevs_id = []

        #for instr in instructions:
        for instr in (i for i in instructions if i.opcode not in ['FREE', 'DISCARD']):

            self._inc += 1
            instr_id    = "%s%d" % (instr.ref(), self._inc)
            instr_style = "%s %s\n" % (instr_id, instr.dot())

            styles  = instr_style
            graph   = ""

            self._rank += 1

            op_ids = []
            for op in instr.operands:

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

                for (ps, cs) in itertools.product( [prev.operands[0]], instr.operands[1:] ):
                    if ps.symbol == cs.symbol:
                        dots += self._edge( op_ids[cs.symbol], prev_id )

            prev    = instr
            prev_id = instr_id

            prevs.append( instr )
            prevs_id.append( instr_id )

        dots += "}"

        return dots

if __name__ == "__main__":

    tracename = ''
    if len(sys.argv) > 1:
        tracename = sys.argv[1]
        
    if os.path.exists(tracename) and os.path.isfile(tracename):
        p = Parser( tracename )
        dotdata = p.dotify_list(p.parse())
        dot = None
        try:
            dot = subprocess.check_call(["which", "-s", "dot"])
        except:
            pass
            
        if dot == 0:
            proc = subprocess.Popen(["dot", "-T", "svg", "-o" + tracename + ".svg"], stdin=subprocess.PIPE)
            proc.communicate(dotdata)
        else:
            print "Could not find 'dot' on your machine"
    else:    
        print "Usage: parser.py <tracefile>"
        
    
