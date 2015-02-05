#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>

using namespace std;

void print_stuff(int values[], int nvalues)
{
    for(int idx=0; idx<nvalues; ++idx) {
        printf("%d", values[idx]);
        if (idx < nvalues-1) {
            printf(", ");
        }
    }
    printf("\n");
}

int prod(int values[], int nvalues)
{
    int res = 1;
    for(int idx=0; idx<nvalues; ++idx) {
        res *= values[idx];
    }
    return res;
}

void calc_weight(int shape[], int weight[], int rank)
{
    int acc = 1;
    for(int idx=rank-1; idx >=0; --idx) {
        weight[idx] = acc;
        acc *= shape[idx];
    }
}

void eidx_to_coord(int eidx, int shape[], int weight[], int rank, int coord[])
{
    for(int idx=0; idx<rank; ++idx) {
        coord[idx] = (eidx / weight[idx]) % shape[idx];
    }
}

int main() {

    int rank = 3;
    int shape[] = {4,4,4};
    int nelements = prod(shape, rank);
    int weight[rank];
    int start_coord[rank];
    int end_coor[rank];
  
    calc_weight(shape, weight, rank);
 
    int nthreads = 3;
    int nrows = prod(shape, rank-1);

    int work_split   = nelements / nthreads;
    int work_spill   = nelements % nthreads;

    for(int tid =0; tid<nthreads; ++tid) {
        int work, begin, end;   // Partition elements
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
        int coord_begin[rank];
        int coord_end[rank];
        int rows_accessed = int(ceil((float(work) / float(shape[rank-1]))));

        eidx_to_coord(begin, shape, weight, rank, coord_begin);
        eidx_to_coord(end, shape, weight, rank, coord_end); 
        
        printf("\nbegin = %d\n",  begin);
        printf("coord_begin = ");
        print_stuff(coord_begin, rank);
        printf("coord_end   = ");
        print_stuff(coord_end, rank);
        printf("end = %d\n",  end);
        printf("rows_accessed = %d\n",  rows_accessed);
        printf("work = %d\n",  work);
    }

    printf("\n** INFO **\n\n");
    printf("nthreads = %d\n",  nthreads);
    printf("rank = %d\n",  rank);
    printf("nelements = %d\n",  nelements);
    printf("shape = ");
    print_stuff(shape, rank);
    printf("spill = %d\n",  work_spill);
    printf("weight = ");
    print_stuff(weight, rank);
    
    return 0;
}
