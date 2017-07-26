from __future__ import print_function
import numpy as np
import random
import operator
import functools


class TYPES:
    NORMAL_INT = ['np.int32', 'np.int64', 'np.uint32', 'np.uint64']
    ALL_INT = NORMAL_INT + ['np.int8', 'np.int16', 'np.uint8', 'np.uint16']
    SIGNED_INT = ['np.int8', 'np.int16', 'np.int32', 'np.int64']
    UNSIGNED_INT = list(set(ALL_INT) - set(SIGNED_INT))
    COMPLEX = ['np.complex64', 'np.complex128']
    FLOAT = ['np.float32', 'np.float64']
    ALL_SIGNED = SIGNED_INT + FLOAT + COMPLEX
    NORMAL = NORMAL_INT + FLOAT
    ALL = ALL_INT + FLOAT + COMPLEX


def gen_shapes(max_ndim, max_dim, iters=0, min_ndim=1):
    for ndim in range(min_ndim, max_ndim + 1):
        shape = [1] * ndim

        if iters:
            # Min shape
            yield shape
            # Max shape
            yield [max_dim] * (ndim)

            for _ in range(iters):
                for d in range(len(shape)):
                    if max_dim == 1:
                        shape[d] = 1
                    else:
                        shape[d] = np.random.randint(1, max_dim)
                yield shape
        else:
            finished = False
            while not finished:
                yield shape
                # Find next shape
                d = ndim - 1
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


def gen_arrays(random_state_name, max_ndim, max_dim=10, min_ndim=1, samples_in_each_ndim=3, dtype="np.float32",
               bh_arg="BH"):
    for shape in gen_shapes(max_ndim, max_dim, samples_in_each_ndim, min_ndim):
        cmd = "%s.random(%s, dtype=%s, bohrium=%s)" % (random_state_name, shape, dtype, bh_arg)
        yield (cmd, shape)


class ViewOfDim:
    def __init__(self, start, step, end):
        self.start = start
        self.step = step
        self.end = end

    def size(self):
        ret = 0
        i = self.start

        assert self.step != 0

        if self.step > 0:
            while i < self.end:
                ret += 1
                i += self.step
        else:
            i += self.step
            while i > self.end:
                ret += 1
                i += self.step
        return ret

    def write(self):
        return "%d:%d:%d" % (self.start, self.end, self.step)


def write_subscription(view):
    ret = "["

    for dim in view[:-1]:
        ret += "%s, " % dim.write()

    ret += "%s]" % view[-1].write()
    return ret


def random_subscription(shape):
    view = []
    view_shape = []

    for dim in shape:
        start = random.randint(0, dim - 1)
        if dim > 3:
            step = random.randint(1, dim // 3)
        else:
            step = 1

        if start + 1 < dim - 1:
            end = random.randint(start + 1, dim - 1)
        else:
            end = start + 1

        # Let's reverse the stride sometimes
        if random.randint(0, 2) == 0:
            (start, end) = (end, start)
            step *= -1

        v = ViewOfDim(start, step, end)
        view.append(v)
        view_shape.append(v.size())
    return write_subscription(view), view_shape


def gen_random_arrays(random_state_name, max_ndim, max_dim=30, min_ndim=1, samples_in_each_ndim=3,
                      dtype="np.float32", bh_arg="BH", no_views=False):
    for cmd, shape in gen_arrays(random_state_name, max_ndim, max_dim, min_ndim, samples_in_each_ndim, dtype, bh_arg):
        yield ("%s" % cmd, shape)

        if functools.reduce(operator.mul, shape) > 1 and not no_views:
            sub_tried = set()

            for _ in range(samples_in_each_ndim):
                sub, vshape = random_subscription(shape)

                if sub not in sub_tried:
                    yield ("%s%s" % (cmd, sub), vshape)
                    sub_tried.add(sub)


def prod(a):
    """Returns the product of the elements in `a`"""
    return functools.reduce(operator.mul, a)
