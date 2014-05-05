#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>

#include <boost/multi_array.hpp>

struct timeval tv;
struct timezone tz;
unsigned long long start, end, delta;


int main (int argc, char **argv) {
  const int size = atoi(argv[1]);
  const int count = atoi(argv[2]);
  const int fullsize = size + 2;

  typedef boost::multi_array<double, 2> array_type;
  typedef array_type::index index;
  array_type full(boost::extents[fullsize][fullsize]);
  array_type work(boost::extents[fullsize][fullsize]);

  // Assign values to the elements
  int values = 0;
  for(index i = 0; i <fullsize; ++i) 
    for(index j = 0; j < fullsize; ++j){
        full[i][j] = values++;
	work[i][j] = 0.0;
    }

  gettimeofday(&tv, &tz);
  start = (long long)tv.tv_usec + ((long long)tv.tv_sec)*1000000;
  
  for(int n=0; n<count; n++)
    {
      //C++ geek must help me with this swat temp=work; work=full; full=temp;
      for(int i=1; i<=size; ++i)
        {
          for(int j=1; j<=size; ++j)
	      work[i][j] = (full[i][j]+full[i-1][j]+full[i+1][j]+full[i][j-1]+full[i][j+1])/5;
        }
    }
  
  gettimeofday(&tv, &tz);
  end = (long long)tv.tv_usec + ((long long)tv.tv_sec)*1000000;
  delta = end - start;
  
  printf("Iter: %d size: %d time: %lf (ANSI C)\n", count, size, delta/(double)1000000.0);

}
