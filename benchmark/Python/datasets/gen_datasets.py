import numpy as np
import point27
import convolve
import wireworld

if __name__ == "__main__":
    benchmarks = ['point27', 'convolve', 'wireworld']
    datasets = {}
    datasets.update(point27.gen_point27_data(400))
    datasets.update(convolve.gen_convolve_data(25))
    datasets.update(wireworld.gen_wireworld_data(1000))

    for filename in datasets:
        print filename
        np.save("/tmp/%s" % filename, datasets[filename])