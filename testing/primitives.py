#!/usr/bin/env python
import numpy as np
import time
import sys
from pprint import pprint as pp

CPHVB=1
if len(sys.argv) > 1:
    CPHVB=int(sys.argv[1])
print "CPHVB state: %d" % CPHVB

bools   = [ np.bool ]
ints    = [ np.int,     np.int8,    np.int16,   np.int32,   np.int64    ]
uints   = [             np.uint8,   np.uint16,  np.uint32,  np.uint64   ]
floats  = [ np.float,                           np.float32, np.float64  ]
complx  = [ np.complex,                                     np.complex64, np.complex128 ]
np_types = bools + ints + uints + floats + complx

ufuncs  = [
    (np.add,            ints+uints+floats),
    (np.subtract,       ints+uints+floats),
    (np.divide,         ints+uints+floats),
    (np.true_divide,    ints+uints+floats),
    (np.floor_divide,   ints+uints+floats),
    (np.multiply,       ints+uints+floats),

    (np.logaddexp,      [np.float32]),
    (np.logaddexp2,     [np.float32]),  

    (np.power,          [np.float32]), 

    (np.remainder,      [np.float32]),
    (np.mod,            [np.float32]),
    (np.fmod,           [np.float32]),

    (np.bitwise_and,    [np.int32]), 
    (np.bitwise_or,     [np.int32]), 
    (np.bitwise_xor,    [np.int32]),

    (np.left_shift,     [np.int32]),    
    (np.right_shift,    [np.int32]),    

    (np.sin, ints+uints+floats),
    (np.cos, ints+uints+floats),
    (np.tan, ints+uints+floats),

    (np.greater,        ints+uints+floats),
    (np.greater_equal,  ints+uints+floats),
    (np.less,           ints+uints+floats), 
    (np.less_equal,     ints+uints+floats),
    (np.not_equal,      ints+uints+floats),
    (np.equal,          ints+uints+floats),
    (np.logical_and,    ints+uints+floats),
    (np.logical_or,     ints+uints+floats),
    (np.logical_xor,    ints+uints+floats),
    (np.logical_not,    ints+uints+floats),
    (np.maximum,        ints+uints+floats),
    (np.minimum,        ints+uints+floats),

    (np.ldexp,      ints+uints+floats),
    (np.negative,   ints+uints+floats),

    (np.invert,         [np.int32]),
    (np.absolute,       ints+floats),

    (np.rint, ints+uints+floats),    
    (np.sign, ints+uints+floats),
    (np.conj, ints+uints+floats),
    (np.exp,    ints+uints+floats),
    (np.exp2,   ints+uints+floats),
    (np.log,    ints+uints+floats),
    (np.log2,   ints+uints+floats),
    (np.log10,  ints+uints+floats),
    (np.log1p,  ints+uints+floats),  
    (np.expm1,  ints+uints+floats),  
    (np.sqrt,   ints+uints+floats),
    (np.square, ints+uints+floats),
    (np.reciprocal, ints+uints+floats),
    (np.ones_like, ints+uints+floats),
    (np.arcsin,     ints+uints+floats),  
    (np.arccos,     ints+uints+floats),  
    (np.arctan,     ints+uints+floats),  
    (np.arctan2,    ints+uints+floats),  
    (np.hypot,      ints+uints+floats),  
    (np.sinh,   ints+uints+floats),  
    (np.cosh,   ints+uints+floats),  
    (np.tanh,   ints+uints+floats),  
    (np.arcsinh, ints+uints+floats), 
    (np.arccosh, ints+uints+floats), 
    (np.arctanh, ints+uints+floats), 
    (np.deg2rad, ints+uints+floats), 
    (np.rad2deg, ints+uints+floats), 
    (np.isfinite,   ints+uints+floats),  
    (np.isinf,      ints+uints+floats),  
    (np.isnan,      ints+uints+floats),  
    (np.signbit,    ints+uints+floats),  
    (np.floor,      ints+uints+floats),  
    (np.ceil,       ints+uints+floats),  
    (np.trunc,      ints+uints+floats),  

    (np.modf, ints+uints+floats),  
    (np.frexp, [np.float32]),  

    #(np.copysign, [np.float32]),   # numpy error
    #(np.nextafter, [np.float32]),  # numpy error

    #(np.isreal, [np.float32]),     # These are not ufuncs!
    #(np.iscomplex, [np.float32])   # These are not ufuncs!
]

def main():
    
    operands = {}
    for np_type in np_types:
        typename = str( np_type.__name__ )
        print typename,
        if typename == 'bool':
            operands[ typename ] = (
                np.array([ True  ] * 1024*1024,    dtype=np_type, dist=CPHVB),
                np.array([ False ] * 1024*1024,    dtype=np_type, dist=CPHVB),
                np.empty([1024*1024], dtype=np_type, dist=CPHVB),
                np.empty([1024*1024], dtype=np_type, dist=CPHVB),
            )
        elif 'int' in typename:
             operands[ typename ] = (
                np.array([ 3  ] * 1024*1024,    dtype=np_type, dist=CPHVB),
                np.array([ 2 ] * 1024*1024,    dtype=np_type, dist=CPHVB),
                np.empty([1024*1024], dtype=np_type, dist=CPHVB),
                np.empty([1024*1024], dtype=np_type, dist=CPHVB),
            )       
        elif 'float' in typename:
             operands[ typename ] = (
                np.array([ 3.75  ] * 1024*1024,    dtype=np_type, dist=CPHVB),
                np.array([ 2 ] * 1024*1024,    dtype=np_type, dist=CPHVB),
                np.empty([1024*1024], dtype=np_type, dist=CPHVB),
                np.empty([1024*1024], dtype=np_type, dist=CPHVB),
            )       
    print ""
    print "Executing %d ufuncs" % len(ufuncs)
    error_count = 0
    for (ufunc, dtypes) in ufuncs:

        for nptype in dtypes:

            ufname = ufunc.__name__ + ','
            print "%s %s,\t" %(ufname+(' '*14)[len(ufname)::], nptype.__name__),
            op1, op2, r1, r2 = operands[ nptype.__name__ ]

            s = time.time()
            invocation_err = ""
            try:

                if ufunc.nin == 2 and ufunc.nout == 1:
                    ufunc( op1, op2, r1 )
                elif ufunc.nin == 1 and ufunc.nout == 1:
                    ufunc( op1, r1 )
                elif ufunc.nin == 1 and ufunc.nout == 2:
                    ufunc( op1, r1, r2 )
                elif ufunc.nin == 1 and ufunc.nout == 0:
                    ufunc( op1 )

            except Exception as e:
                invocation_err = str(e)
                error_count += 1

            val = r1[0] if str(r1[0]) else '?';
            print "%f,\t%s,\t[%s]." % (time.time() -s, val, invocation_err)

    print "%d successful invocations %s with error." % (len(ufuncs)-error_count, error_count)

if __name__ == "__main__":
    main()
