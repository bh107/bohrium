#include <iostream>
#include "bh/bh.hpp"

using namespace std;
using namespace bh;

template <typename T>
multi_array<T>& iterate(int w, int h, int i)
{
    multi_array<T> grid(h+2,w+2), center, north, south, east, west;

    grid = 0.0;
    grid[_(0,h,1)][ 0] = -273.15;
    grid[_(0,h,1)][-1] = -273.15;
    grid[-1][_(0,w,1)] = -273.15;
    grid[ 0][_(0,w,1)] =    40.0;

    center  = grid[_(1,-1,1)][_(1,-1,1)];
    north   = grid[_(0,-2,1)][_(1,-1,1)];
    south   = grid[_(2, h,1)][_(1,-1,1)];
    east    = grid[_(1,-1,1)][_(2, w,1)];
    west    = grid[_(1,-1,1)][_(0,-2,1)];

    for(int k=0; k<i; k++) {
        center = (T)0.2*(center+north+east+west+south);
    }

    return grid;
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

