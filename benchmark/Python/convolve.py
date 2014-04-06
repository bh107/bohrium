import util
import bohrium as np
from bohrium.stdviews import cartesian

def convolve_init(image_fn='/tmp/bigimage.npy', filter_fn='/tmp/filter.npy', bohrium=True):
    image           = np.load(iname, bohrium=bohrium)
    image_filter    = np.load(fname, bohrium=False)

    return (image, image_filter)

def convolve(image, image_filter):
    """TODO: Describe the benchmark."""
    
    views   = cartesian(image, len(image_filter))
    result  = sum(d[0]*d[1] for d in zip(views, image_filter.flatten()))

    return result

if __name__ = "__main__":    
    B = util.Benchmark()
    image, image_filter = convolve_init(bohrium=B.bohrium)
    B.start()
    result = convolve(image, image_filter)
    B.stop()
    B.pprint()
