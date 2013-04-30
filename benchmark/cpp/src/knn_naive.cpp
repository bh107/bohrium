#include <iostream>
#include "bh/bh.hpp"

using namespace std;
using namespace bh;

template <typename T>
multi_array<T>& knn_naive(int n, int features, int k)
{
    multi_array<T> data_set((features, n)),
                    target(features),
                    sqd;

    sqd = sqrt((data_set - target)*(data_set - target)).reduce(0);

    return sqd;
}

int main(int argc, char* argv[])
{
    int height, width, iterations;
    if (argc < 4) {
        cout << "Error: Not enough argumnts, call like: " << endl;
        cout << argv[0] << " 600 400 2" << endl;
        cout << "For a 600x400 grid and 2 iterations." << endl;
        return 0;
    }
    width       = atoi(argv[1]);
    height      = atoi(argv[2]);
    iterations  = atoi(argv[3]);

    cout << "[Jacobi Stencil(" \
            "width=" << width << \
            ",height=" << height << \
            ",iter=" << iterations << \
            ")]" << endl;
    iterate<double>(width, height, iterations);
    cout << "done!" << endl;

    return 0;
}

