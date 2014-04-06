import util
import bohrium as np
from bohrium.stdviews import D2P8, no_boarder

def wireworld_init(fname='/tmp/wireworld.npy', use_bohrium=True):
    return np.load(fname, bohrium=use_bohrium)

def wireworld(world, iterations):
    """TODO: Describe the benchmark."""

    sim = no_boarder(world ,1)  #Active Machine
    stencil = D2P8(world)       #Stencil for counting heads
    for _ in xrange(iterations):
        NC = sum([v==2 for v in stencil]) #Count number of head neighbors
        #Mask conductor->head
        MASK = ((NC==1) & (sim==8)) | ((NC==2) & (sim==8))
        sim *= ~MASK    #New head pos->0
        sim += MASK * 1 #New head pos->1
        MASK = (sim==8) #Mask non conductors
        sim *= ~MASK    #conductors->0
        sim += MASK * 4 #conductors->4   
        sim *= 2        #Upgrade all to new state

    return sim

if __name__ == "__main__":
    B = util.Benchmark()
    (I,) = B.size
    world = wireworld_init(use_bohrium=B.bohrium)
    B.start()
    result = wireworld(world, I)
    B.stop()
    B.pprint()
