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

  double **full = malloc(fullsize);//Input and output array
  for(i=0; i<fullsize;i++){
    full[i]=(double *)malloc(fullsize*sizeof(double));
    for(j=0;j<fullsize;j++)full[i][j]=1.0*i*j;
  }

  double **work = malloc(mem_size);//Tmp array
  for(i=0; i<fullsize;i++){
    work[i]=(double *)malloc(fullsize*sizeof(double));
    for(j=0;j<fullsize;j++)work[i][j]=0.0;
  }
  double **temp;
  gettimeofday(&tv, &tz);
  start = (long long)tv.tv_usec + ((long long)tv.tv_sec)*1000000;
  

  for(n=0; n<count; n++)
    {
      temp=work; work=full; full=temp;
      for(i=1; i<=size; ++i)
        {
          for(j=1; j<=size; ++j)
            {
	      work[i][j] = (full[i][j]+full[i-1][j]+full[i+1][j]+full[i][j-1]+full[i][j+1])/5;
            }
        }
    }
  
  gettimeofday(&tv, &tz);
  end = (long long)tv.tv_usec + ((long long)tv.tv_sec)*1000000;
  delta = end - start;
  
  printf("Iter: %d size: %d time: %lf (ANSI C)\n", count, size, delta/(double)1000000.0);
  
  //free(full);
  //free(work);
}  
