#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <mpi.h>

struct timeval tv;
struct timezone tz;
unsigned long long start, end, delta;
typedef void (*benchmark)(int height, int width, double *grid, int iter);

void sequential(int height, int width, double *grid, int iter)
{
  double *T = malloc(height*width*sizeof(double));
  for(int n=0; n<iter; ++n)
  {
    double *a = grid;
    double *t = T;
    double delta=0;
    for(int i=1; i<height-1; ++i)
    {
      double *up     = a+1;
      double *left   = a+width;
      double *right  = a+width+2;
      double *down   = a+1+width*2;
      double *center = a+width+1;
      double *t_center = t+width+1;
      for(int j=0; j<width-2; ++j)
      {
        *t_center = (*center + *up + *left + *right + *down) / 5.0;
        delta += fabs(*t_center + *center);
        center++;up++;left++;right++;down++;t_center++;
      }
      a += width;
      t += width;
    }
    memcpy(grid, T, height*width*sizeof(double));
  }
}

void innerloop(double *grid, double *T, int width, int i, double *delta)
{
    int a = i * width;
    double *up     = &grid[a+1];
    double *left   = &grid[a+width];
    double *right  = &grid[a+width+2];
    double *down   = &grid[a+1+width*2];
    double *center = &grid[a+width+1];
    double *t_center = &T[a+width+1];
    for(int j=0; j<width-2; ++j)
    {
        *t_center = (*center + *up + *left + *right + *down) / 5.0;
        *delta += fabs(*t_center + *center);
        center++;up++;left++;right++;down++;t_center++;
    }
}

void openmp(int height, int width, double *grid, int iter)
{
    double *T = malloc(height*width*sizeof(double));
    for(int n=0; n<iter; ++n)
    {
        double delta=0;
        #pragma omp parallel for shared(grid,T) reduction(+:delta)
        for(int i=0; i<height-2; ++i)
        {
            innerloop(grid, T, width, i, &delta);
        }
        memcpy(grid, T, height*width*sizeof(double));
    }
}

void openmp_mpi(int height, int width, double *grid, int iter)
{
    int myrank, worldsize;
    MPI_Comm comm;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldsize);

    int periods[] = {0};
    MPI_Cart_create(MPI_COMM_WORLD, 1, &worldsize,
                    periods, 1, &comm);

    double *T = malloc(height*width*sizeof(double));
    for(int n=0; n<iter; ++n)
    {
        int p_src, p_dest;
        //Send/receive - neighbor above
        MPI_Cart_shift(comm,0,1,&p_src,&p_dest);
        MPI_Sendrecv(grid+width,width,MPI_DOUBLE,
                     p_dest,1,
                     grid,width, MPI_DOUBLE,
                     p_src,1,comm,MPI_STATUS_IGNORE);
        //Send/receive - neighbor below
        MPI_Cart_shift(comm,0,-1,&p_src,&p_dest);
        MPI_Sendrecv(grid+(height-2)*width, width,MPI_DOUBLE,
                     p_dest,1,
                     grid+(height-1)*width,
                     width,MPI_DOUBLE,
                     p_src,1,comm,MPI_STATUS_IGNORE);

        double delta=0, global_delta;
        #pragma omp parallel for shared(grid,T) reduction(+:delta)
        for(int i=0; i<height-2; ++i)
        {
            innerloop(grid, T, width, i, &delta);
        }
        memcpy(grid, T, height*width*sizeof(double));
        MPI_Allreduce(&global_delta, &delta, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }
}


int nbenchs = 1;
benchmark func[] = {&sequential, &openmp, &openmp_mpi};
const char *name[] = {"Sequential", "OpenMP", "OpenMP+MPI"};

int main()
{
    int iter = 10;
    int width = 4998+2; // Size + borders.

    MPI_Init(NULL,NULL);
    int myrank, worldsize;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldsize);

    for(int i=0; i<nbenchs; i++)
    {
        int height = width;
        if(i>1)//A MPI benchmark
        {
            MPI_Barrier(MPI_COMM_WORLD);
            //Local vertical size. NB: the horizontal size is always the full grid including borders
            height = (width-2) / worldsize;
            if(myrank == worldsize-1)
                height += (width-2) % worldsize;
            height += 2;//Add a ghost line above and below
        }
        else if(myrank != 0)
            continue;

        double *grid = malloc(height*width*sizeof(double));

        for(int j=0; j<height*width;j++)
            grid[j] = j;

        gettimeofday(&tv, &tz);
        start = (long long)tv.tv_usec + ((long long)tv.tv_sec)*1000000;

        func[i](height,width,grid,iter);

        gettimeofday(&tv, &tz);
        end = (long long)tv.tv_usec + ((long long)tv.tv_sec)*1000000;
        delta = end - start;

        if(myrank == 0)
            printf("%s - iter: %d size: %d time: %lf\n", name[i], iter, width, delta/(double)1000000.0);

        free(grid);
    }

    MPI_Finalize();
    return 0;
}
