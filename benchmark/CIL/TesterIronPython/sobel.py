import os
import sys
import time

#from numpy import fromfile, uint16, uint32, int64, array, sqrt, isnan, rint
from numcil import fromfile, uint16, uint32, int64, array, sqrt, rint
from cphhpc.signal.convolution import convolve2d

# http://homepages.inf.ed.ac.uk/rbf/HIPR2/sobel.htm
def sobel(input, data_type):
    sobel_window_x = array([[-1, 0, 1], 
                            [-2, 0, 2],
                            [-1, 0, 1]]).astype(data_type)
                     
    sobel_window_y = array([[-1, -2, -1], 
                            [0, 0, 0],
                            [1, 2, 1]]).astype(data_type)
     
    sobel_x = convolve2d(input, sobel_window_x, out=None, data_type=data_type)
    sobel_y = convolve2d(input, sobel_window_y, out=None, data_type=data_type)

    # Approx
    # from numpy import abs
    # result = abs(sobel_x) + abs(sobel_y)
    result = sqrt(sobel_x**2 + sobel_y**2)
    
    return result 

def main():
    argc = len(sys.argv)-1
    
    if argc != 2:
        print "USAGE: %s input_file output_file" % (sys.argv[0])
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    input_data_type = uint16
    output_data_type = uint32
    sobel_data_type = int64
    
    input = fromfile(input_file, dtype=input_data_type)
    input.shape = (2048, 2048)

    t1 = time.time()
    result = sobel(input, sobel_data_type)
    t2 = time.time()
    print "Sobel time: %s secs" % (t2-t1)
        
    output = (result).astype(output_data_type)
    output.tofile(output_file)

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "Spa_mid_20x_1a0001.rec.16bit.raw", "out.raw"]
    main()
