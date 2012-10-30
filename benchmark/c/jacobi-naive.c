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
  

#define WORK(x,y) work[((y)*size)+x]
#define FULL(x,y) full[((y)*size)+x]

  for(n=0; n<count; n++)
    {
      temp=work; work=full; full=temp;
      for(i=1; i<=size; ++i)
        {
          for(j=1; j<=size; ++j)
            {
	      FULL(i,j) = (WORK(i,j)+WORK(i-1,j)+WORK(i+1,j)+WORK(i,j-1)+WORK(i,j+1))/5;
            }
        }
    }
  
  gettimeofday(&tv, &tz);
  end = (long long)tv.tv_usec + ((long long)tv.tv_sec)*1000000;
  delta = end - start;
  
  printf("Iter: %d size: %d time: %lf (ANSI C)\n", count, size, delta/(double)1000000.0);
  
  free(full);
  free(work);
}  
