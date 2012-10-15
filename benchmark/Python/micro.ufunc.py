#!/usr/bin/env python
import cphvbnumpy as np
import cphvbbridge as cb
import json
import time
import sys
import util
import tempfile


ttup = [
    ('CPHVB_BOOL',   np.bool),

    ('CPHVB_INT8',   np.int8),
    ('CPHVB_INT16',  np.int16),
    ('CPHVB_INT32',  np.int32),
    ('CPHVB_INT64',  np.int64),

#    ('CPHVB_FLOAT16',  np.float16),
    ('CPHVB_FLOAT32',  np.float32),
    ('CPHVB_FLOAT64',  np.float64),

    ('CPHVB_UINT8',   np.uint8),
    ('CPHVB_UINT16',  np.uint16),
    ('CPHVB_UINT32',  np.uint32),
    ('CPHVB_UINT64',  np.uint64),

    ('CPHVB_COMPLEX64', np.complex64),
    ('CPHVB_COMPLEX128', np.complex128)
]
tmap = dict(ttup)

def main( B, runs=5 ):

    N = B.size.pop()

    print "Loading cphvb-opcodes."
    instructions    = json.load(open('../../core/codegen/opcodes.json'))
    ufuncs          = [ufunc for ufunc in instructions if not ufunc['system_opcode']]

    ignore_t = ['CPHVB_COMPLEX64']
    ignore_f = ['CPHVB_IDENTITY']
    
    print "Allocating operands."
    operands = {}                   # Setup operands of various types
    for cphvb_type, np_type in ttup:

        if 'bool' in cphvb_type.lower():
            operands[ cphvb_type ] = (
                np.ones([N],            dtype=np_type, cphvb=B.cphvb),
                np.array([ True  ] * N, dtype=np_type, cphvb=B.cphvb),
                np.array([ False ] * N, dtype=np_type, cphvb=B.cphvb),
            )
        elif 'int' in cphvb_type.lower():
            operands[ cphvb_type ] = (
                np.ones([N],         dtype=np_type, cphvb=B.cphvb),
                np.array([ 3 ] * N,  dtype=np_type, cphvb=B.cphvb),
                np.array([ 2 ] * N,  dtype=np_type, cphvb=B.cphvb),
            )
        elif 'float' in cphvb_type.lower():
            operands[ cphvb_type ] = (
                np.ones([N],             dtype=np_type, cphvb=B.cphvb),
                np.array([ 3.75 ] * N,   dtype=np_type, cphvb=B.cphvb),
                np.array([ 2.0  ] * N,   dtype=np_type, cphvb=B.cphvb),
            )
        elif 'complex' in cphvb_type.lower():
            operands[ cphvb_type ] = (
                np.ones([N],             dtype=np_type, cphvb=B.cphvb),
                np.array([ 3.75 ] * N,   dtype=np_type, cphvb=B.cphvb),
                np.array([ 2.0  ] * N,   dtype=np_type, cphvb=B.cphvb),
            )
    cb.flush()
    
    print ""
    print "Executing %d ufuncs" % len(ufuncs)
    error_count = 0
    results     = []
    for ufunc in ufuncs:

        opcode  = ufunc['opcode']
        if opcode in ignore_f:
            continue

        types   = ufunc['types']
        nop     = ufunc['nop']
        fp      = np.__dict__[opcode.replace('CPHVB_','').lower()]

        for typesig in types:

            params = []
            if nop == 2:
                params.append( operands[typesig[1]][1] )
                params.append( operands[typesig[0]][0] )
            elif nop == 3:
                params.append( operands[typesig[1]][1] )
                params.append( operands[typesig[2]][2] )
                params.append( operands[typesig[0]][0] )
            else:
                print "WHAT!!!? "+nop

            invocation_err = ""
            times = []
            for _ in xrange(0, runs):
                s = elapsed = 0.0
                try:
                    cb.flush()
                    s = time.time()
                    fp( *params )
                    cb.flush()
                    elapsed = time.time() - s
                    times.append( elapsed )
                except Exception as e:
                    invocation_err = str(e)
                    error_count += 1

            val = params[-1][0] if str(params[-1][0]) else '?'
            results.append( [opcode, typesig, times, str(val), invocation_err, B.cphvb] )

    print "%d successful invocations %s with error." % (len(ufuncs)-error_count, error_count)

    return results

if __name__ == "__main__":


    B = util.Benchmark()
    B.start()
    results = main( B, 5 )
    B.stop()
    B.pprint()

    with tempfile.NamedTemporaryFile(delete=False, dir='/tmp', prefix='res-', suffix='.json') as fd:
        json.dump(results, fd, indent=4)

