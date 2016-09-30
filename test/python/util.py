from __future__ import print_function
import numpy as np
import random
import operator


def gen_shapes(max_ndim, max_dim, iters=0, min_ndim=1):
    for ndim in xrange(min_ndim,max_ndim+1):
        shape = [1]*ndim
        if iters:
            yield shape #Min shape
            yield [max_dim]*(ndim) #Max shape
            for _ in xrange(iters):
                for d in xrange(len(shape)):
                    shape[d] = np.random.randint(1,max_dim)
                yield shape
        else:
            finished = False
            while not finished:
                yield shape
                #Find next shape
                d = ndim-1
                while True:
                    shape[d] += 1
                    if shape[d] > max_dim:
                        shape[d] = 1
                        d -= 1
                        if d < 0:
                            finished = True
                            break
                    else:
                        break


def gen_arrays(random_state_name, max_ndim, max_dim=10, min_ndim=1, samples_in_each_ndim=3, dtype="np.float32", bh_arg="BH"):
    for shape in gen_shapes(max_ndim, max_dim, samples_in_each_ndim, min_ndim):
        cmd = "%s.random(%s, dtype=%s, bohrium=%s)" % (random_state_name, shape, dtype, bh_arg)
        yield (cmd, shape)


class ViewOfDim:
    def __init__(self, start, step, end):
        self.start = start
        self.step = step
        self.end = end

    def write(self):
        return "%d:%d:%d"%(self.start, self.end, self.step)


def write_subscription(view):
    ret = "["
    for dim in view[:-1]:
        ret += "%s, "%dim.write()
    ret += "%s]"%view[-1].write()
    return ret


def random_subscription(shape):
    view = []
    for dim in shape:
        start = random.randint(0, dim-1)
        if dim > 3:
            step = random.randint(1, dim/2)
        else:
            step = 1
        if start+1 < dim-1:
            end = random.randint(start+1, dim-1)
        else:
            end = start+1
        view.append(ViewOfDim(start, step, end))
    return write_subscription(view)


def gen_random_arrays(random_state_name, max_ndim, max_dim=10, min_ndim=1, samples_in_each_ndim=3, dtype="np.float32", bh_arg="BH"):
    for cmd, shape in gen_arrays(random_state_name, max_ndim, max_dim, min_ndim, samples_in_each_ndim, dtype, bh_arg):
        yield ("%s" % cmd, len(shape))
        if reduce(operator.mul, shape) > 1:
            sub_tried = set()
            for _ in range(samples_in_each_ndim):
                sub = random_subscription(shape)
                if sub not in sub_tried:
                    yield ("%s%s" % (cmd, random_subscription(shape)), len(shape))
                    sub_tried.add(sub)
