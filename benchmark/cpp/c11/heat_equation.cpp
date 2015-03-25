#include <iostream>
#include <sstream>
#include <string>
#include <cmath>

using namespace std;

int main(){
    const int xdim = 10000;
    const int ydim = 10000;
    auto grid = new double[ydim][xdim];

    double epsilon = 0.005;
    double delta = epsilon+1.0;
    auto temp = new double[ydim][xdim];

    for(int i=0; i<ydim; i++){
        for(int j=0;j<xdim;j++){
            grid[i][j] = 0;
        }
    }
    for(int i=0; i<ydim; i++){
        grid[i][0]      = -273.15;
        grid[i][xdim-1] = -273.15;
    }
    for(int i=0; i<xdim; i++){
        grid[0][i]      = -273.15;
        grid[ydim-1][i] = 40.0;
    }

    auto iterations = 0;
    while(delta>epsilon) {
        ++iterations;
        delta = 0.0;
        for(int i=1; i<ydim-1; i++){
            #pragma omp parallel for reduction(+:delta)
            for(int j=1;j<xdim-1;j++){
                temp[i][j] = (grid[i-1][j] + grid[i+1][j] + grid[i][j] + grid[i][j-1] + grid[i][j+1])*0.2;
                delta += abs(temp[i][j] - grid[i][j]);
            }
        }

        #pragma omp parallel for collapse(2)
        for(int i=1;i<ydim-1; i++){
            for(int j=1;j<xdim-1;j++){
                grid[i][j] = temp[i][j];
            }
        }
        if (iterations == 10) {
            break;
        }
    }
    cout << "Iterations " << iterations << " GRID" << grid[500][500] << endl;
    return 0;
}
