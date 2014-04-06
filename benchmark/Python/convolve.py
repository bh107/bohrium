import util
import bohrium as np
from bohrium.stdviews import cartesian

def convolve_init(image_fn='bigimage.npy', filter_fn='filter.npy', bohrium=True):
    image   = np.load(iname, bohrium=bohrium)
    filter  = np.load(fname, bohrium=False)

    return (image, filter)

def convolve(image, filter):
    """TODO: Describe the benchmark."""
    
    views = cartesian(image, len(filter))
    result = sum(d[0]*d[1] for d in zip(views, filter.flatten()))

    return result

if __name__ = "__main__":    
    B = util.Benchmark()
    image, filter = convolve_init(bohrium=B.bohrium)
    B.start()
    result = convolve(image, filter, bohrium=B.bohrium)
    B.stop()
    B.pprint()