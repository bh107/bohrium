#include <stdlib.h>
#include <iostream>
#include <omp.h>
#include <math.h>

using namespace std;

void print_stuff(size_t values[], size_t nvalues)
{
    for(size_t idx=0; idx<nvalues; ++idx) {
        cout << values[idx];
        if (idx < nvalues-1) {
            cout << ", ";
        }
    }
    cout << endl;
}

size_t prod(size_t values[], size_t nvalues)
{
    size_t res = 1;
    for(int idx=0; idx<nvalues; ++idx) {
        res *= values[idx];
    }
    return res;
}

void calc_weight(size_t shape[], size_t weight[], size_t rank)
{
    size_t acc = 1;
    for(int idx=rank-1; idx >=0; --idx) {
        weight[idx] = acc;
        acc *= shape[idx];
    }
}

void eidx_to_coord(size_t eidx, size_t shape[], size_t weight[], size_t rank, size_t coord[])
{
    for(int idx=0; idx<rank; ++idx) {
        coord[idx] = (eidx / weight[idx]) % shape[idx];
    }
}

int main() {

    size_t rank = 3;
    size_t shape[] = {4,4,4};
    size_t nelements = prod(shape, rank);
    size_t weight[rank];
    size_t start_coord[rank];
    size_t end_coor[rank];
  
    calc_weight(shape, weight, rank);
 
    size_t nthreads = 3;
    size_t nrows = prod(shape, rank-1);

    size_t work_split   = nelements / nthreads;
    size_t work_spill   = nelements % nthreads;

    for(size_t tid =0; tid<nthreads; ++tid) {
        size_t work, begin, end;   // Partition elements
        //tid = omp_get_thread_num();
        if (tid < work_spill) {
            work = work_split + 1;
            begin = tid * work;
        } else {
            work = work_split;
            begin = tid * work + work_spill;
        }
        end = begin + work -1; 
                                        // Convert to loop-boundaries
        size_t coord_begin[rank];
        size_t coord_end[rank];
        size_t rows_accessed = size_t(ceil((float(work) / float(shape[rank-1]))));

        eidx_to_coord(begin, shape, weight, rank, coord_begin);
        eidx_to_coord(end, shape, weight, rank, coord_end); 
        //cout << "TID: " << tid << endl;
        cout << "begin = " << begin << endl;
        cout << "coord_begin: ";
        print_stuff(coord_begin, rank);
        cout << "coord_end: ";
        print_stuff(coord_end, rank);
        cout << "end = " << end << endl;
        cout << "rows_accessed: " << rows_accessed  << endl;
        cout << "work:" << work << endl << endl;
    }
    cout << "nthreads = " << nthreads << endl;
    cout << "rank = " << rank << endl;
    cout << "nelements = " << nelements << endl;
    cout << "shape = ";
    print_stuff(shape, rank);
    cout << "spill = " << work_spill << endl;
    cout << "weight = ";
    print_stuff(weight, rank);
    
    return 0;
}
