#!/usr/bin/env python
import numpy as np
import unittest

bools   = [ np.bool ]
ints    = [ np.int,     np.int8,    np.int16,   np.int32,   np.int64    ]
uints   = [             np.uint8,   np.uint16,  np.uint32,  np.uint64   ]
floats  = [ np.float,                           np.float32, np.float64  ]
complx  = [ np.complex,                                     np.complex64, np.complex128 ]
np_types = bools + ints + uints + floats + complx

class TestPrimitives(unittest.TestCase):

    def setUp(self):

        self.operands   = {'score':{}, 'numpy':{}}
        backends        = {'score':True, 'numpy':False}
        for backend in backends:
            for np_type in np_types:
                typename = str( np_type.__name__ )
                if typename == 'bool':
                    self.operands[backend][ typename ] = (
                        np.array([ True  ] * 1024,    dtype=np_type, dist=backends[backend]),
                        np.array([ False ] * 1024,    dtype=np_type, dist=backends[backend]),
                        np.empty([1024], dtype=np_type, dist=backends[backend]),
                        np.empty([1024], dtype=np_type, dist=backends[backend])
                    )
                elif 'int' in typename:
                     self.operands[backend][ typename ] = (
                        np.array([ -1  ] * 1024,    dtype=np_type, dist=backends[backend]),
                        np.array([ 1 ] * 1024,    dtype=np_type, dist=backends[backend]),
                        np.empty([1024], dtype=np_type, dist=backends[backend]),
                        np.empty([1024], dtype=np_type, dist=backends[backend])
                    )       
                elif 'float' in typename:
                     self.operands[backend][ typename ] = (
                        np.array([ -1.0  ] * 1024,    dtype=np_type, dist=backends[backend]),
                        np.array([ 1.0 ] * 1024,    dtype=np_type, dist=backends[backend]),
                        np.empty([1024], dtype=np_type, dist=backends[backend]),
                        np.empty([1024], dtype=np_type, dist=backends[backend])
                    )

    def test_add(self):
    
        (ac, bc, rc, _) = self.operands['score']['int']
        np.add( ac, bc, rc )
    
        (an, bn, rn, _) = self.operands['numpy']['int']
        np.add( an, bn, rn )
    
        for i in xrange(0,len(rn)):
            self.assertEqual(rc[i], rn[i])
    
    def test_sub(self):
    
        (ac, bc, rc, _) = self.operands['score']['int']
        np.subtract( ac, bc, rc )
    
        (an, bn, rn, _) = self.operands['numpy']['int']
        np.subtract( an, bn, rn )
    
        for i in xrange(0,len(rn)):
            self.assertEqual(rc[i], rn[i])
    
    def test_divide(self):
    
        (ac, bc, rc, _) = self.operands['score']['int']
        np.divide( ac, bc, rc )
    
        (an, bn, rn, _) = self.operands['numpy']['int']
        np.divide( an, bn, rn )
    
        for i in xrange(0,len(rn)):
            self.assertEqual(rc[i], rn[i])
    
    def test_true_divide(self):
    
        (ac, bc, rc, _) = self.operands['score']['int']
        np.true_divide( ac, bc, rc )
    
        (an, bn, rn, _) = self.operands['numpy']['int']
        np.true_divide( an, bn, rn )
    
        for i in xrange(0,len(rn)):
            self.assertEqual(rc[i], rn[i])
    
    def test_floor_divide(self):
    
        (ac, bc, rc, _) = self.operands['score']['int']
        np.floor_divide( ac, bc, rc )
    
        (an, bn, rn, _) = self.operands['numpy']['int']
        np.floor_divide( an, bn, rn )
    
        for i in xrange(0,len(rn)):
            self.assertEqual(rc[i], rn[i])
    
    def test_multiply(self):
    
        (ac, bc, rc, _) = self.operands['score']['int']
        np.multiply( ac, bc, rc )
    
        (an, bn, rn, _) = self.operands['numpy']['int']
        np.multiply( an, bn, rn )
    
        for i in xrange(0,len(rn)):
            self.assertEqual(rc[i], rn[i])
    
    def test_logaddexp(self):
    
        (ac, bc, rc, _) = self.operands['score']['int']
        np.logaddexp( ac, bc, rc )
    
        (an, bn, rn, _) = self.operands['numpy']['int']
        np.logaddexp( an, bn, rn )
    
        for i in xrange(0,len(rn)):
            self.assertEqual(rc[i], rn[i])
    
    def test_logaddexp2(self):
    
        (ac, bc, rc, _) = self.operands['score']['int']
        np.logaddexp2( ac, bc, rc )
    
        (an, bn, rn, _) = self.operands['numpy']['int']
        np.logaddexp2( an, bn, rn )
    
        for i in xrange(0,len(rn)):
            self.assertEqual(rc[i], rn[i])
    
    def test_power(self):
   
        (ac, bc, rc, _) = self.operands['score']['float']
        np.power( ac, rc )
    
        (an, bn, rn, _) = self.operands['numpy']['float']
        np.power( an, rn )
    
        for i in xrange(0,len(rn)):
            self.assertEqual(rc[i], rn[i])
    
    def test_remainder(self):
    
        (ac, bc, rc, _) = self.operands['score']['int']
        np.remainder( ac, bc, rc )
    
        (an, bn, rn, _) = self.operands['numpy']['int']
        np.remainder( an, bn, rn )
    
        for i in xrange(0,len(rn)):
            self.assertEqual(rc[i], rn[i])
    
    def test_fmod(self):
    
        (ac, bc, rc, _) = self.operands['score']['int']
        np.fmod( ac, bc, rc )
    
        (an, bn, rn, _) = self.operands['numpy']['int']
        np.fmod( an, bn, rn )
    
        for i in xrange(0,len(rn)):
            self.assertEqual(rc[i], rn[i])
    
    def test_left_shift(self):
    
        (ac, bc, rc, _) = self.operands['score']['int']
        np.left_shift( ac, bc, rc )
    
        (an, bn, rn, _) = self.operands['numpy']['int']
        np.left_shift( an, bn, rn )
    
        for i in xrange(0,len(rn)):
            self.assertEqual(rc[i], rn[i])

    def test_right_shift(self):

        (ac, bc, rc, _) = self.operands['score']['int']
        np.right_shift( ac, bc, rc )

        (an, bn, rn, _) = self.operands['numpy']['int']
        np.right_shift( an, bn, rn )

        for i in xrange(0,len(rn)):
            self.assertEqual(rc[i], rn[i])

    def test_bitwise_and(self):
    
        (ac, bc, rc, _) = self.operands['score']['int']
        np.bitwise_and( ac, bc, rc )
    
        (an, bn, rn, _) = self.operands['numpy']['int']
        np.bitwise_and( an, bn, rn )
    
        for i in xrange(0,len(rn)):
            self.assertEqual(rc[i], rn[i])
    
    def test_bitwise_or(self):
    
        (ac, bc, rc, _) = self.operands['score']['int']
        np.bitwise_or( ac, bc, rc )
    
        (an, bn, rn, _) = self.operands['numpy']['int']
        np.bitwise_or( an, bn, rn )
    
        for i in xrange(0,len(rn)):
            self.assertEqual(rc[i], rn[i])
    
    def test_bitwise_xor(self):
    
        (ac, bc, rc, _) = self.operands['score']['int']
        np.bitwise_xor( ac, bc, rc )
    
        (an, bn, rn, _) = self.operands['numpy']['int']
        np.bitwise_xor( an, bn, rn )
    
        for i in xrange(0,len(rn)):
            self.assertEqual(rc[i], rn[i])
    
    def test_sin(self):
    
        (ac, bc, rc, _) = self.operands['score']['int']
        np.sin( ac, rc )
    
        (an, bn, rn, _) = self.operands['numpy']['int']
        np.sin( an, rn )
    
        for i in xrange(0,len(rn)):
            self.assertEqual(rc[i], rn[i])
    
    def test_cos(self):
    
        (ac, bc, rc, _) = self.operands['score']['int']
        np.cos( ac, rc )
    
        (an, bn, rn, _) = self.operands['numpy']['int']
        np.cos( an, rn )
    
        for i in xrange(0,len(rn)):
            self.assertEqual(rc[i], rn[i])
    
    def test_tan(self):
    
        (ac, bc, rc, _) = self.operands['score']['int']
        np.tan( ac, rc )
    
        (an, bn, rn, _) = self.operands['numpy']['int']
        np.tan( an, rn )
    
        for i in xrange(0,len(rn)):
            self.assertEqual(rc[i], rn[i])

    def test_rad2deg(self):
    
        (ac, bc, rc, _) = self.operands['score']['int']
        np.rad2deg( ac, rc )
    
        (an, bn, rn, _) = self.operands['numpy']['int']
        np.rad2deg( an, rn )
    
        for i in xrange(0,len(rn)):
            self.assertEqual(rc[i], rn[i])
            
    def test_deg2rad(self):
    
        (ac, bc, rc, _) = self.operands['score']['int']
        np.deg2rad( ac, rc )
    
        (an, bn, rn, _) = self.operands['numpy']['int']
        np.deg2rad( an, rn )
    
        for i in xrange(0,len(rn)):
            self.assertEqual(rc[i], rn[i])

    def test_ldexp(self):
   
        (aci, bci, rci, _) = self.operands['score']['int']
        (acf, bcf, rcf, _) = self.operands['score']['float']
        np.ldexp( acf, bci, rcf )
    
        (ani, bni, rni, _) = self.operands['numpy']['int']
        (anf, bnf, rnf, _) = self.operands['numpy']['float']
        np.ldexp( anf, bni, rnf )
    
        for i in xrange(0,len(rnf)):
            self.assertEqual(rcf[i], rnf[i])

if __name__ == "__main__":
    unittest.main()        
