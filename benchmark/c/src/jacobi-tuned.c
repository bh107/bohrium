#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>
struct timeval tv;
struct timezone tz;
unsigned long long start, end, delta;


int main(int argc, char *argv[])
{
  const int size = atoi(argv[1]);
  const int count = atoi(argv[2]);
  const int fullsize = size + 2;
  const int mem_size = (fullsize * fullsize * sizeof(double));

  int n,i,j;

  double *full = malloc(mem_size);//Input and output array
  bzero(full, mem_size);
  for(i=0;i<fullsize*fullsize;i++)full[i]=1.0*i;
  
  double *work = malloc(mem_size);//Tmp array
  bzero(work,mem_size);
  
  double *temp;
  gettimeofday(&tv, &tz);
  start = (long long)tv.tv_usec + ((long long)tv.tv_sec)*1000000;
  
  for(n=0; n<count; n++)
    {
      temp=work; work=full; full=temp;
      double *a = full;
      double *w = work;
      for(i=0; i<size; ++i)
        {
	  double *up    = a+1;
	  double *left  = a+fullsize;
	  double *right = a+fullsize+2;
	  double *down  = a+1+fullsize*2;
	  double *wp=w+fullsize+1;
	  
	  for(j=0; j<size; ++j)
	      *wp = (*wp++ + *up++ + *left++ + *right++ + *down++)*0.2;
	  a += fullsize;
	  w += fullsize;
        }
    }
  
  gettimeofday(&tv, &tz);
  end = (long long)tv.tv_usec + ((long long)tv.tv_sec)*1000000;
  delta = end - start;
  
  printf("Iter: %d size: %d time: %lf (ANSI C)\n", count, size, delta/(double)1000000.0);
  
  free(full);
  free(work);

  return 0;
}
