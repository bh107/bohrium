/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with Bohrium. 

If not, see <http://www.gnu.org/licenses/>.
*/

#include "timing.h"

#ifdef _WIN32
int gettimeofday(timeval *tp, void *tzp) { 
    LARGE_INTEGER freq;
    LARGE_INTEGER s1;
    QueryPerformanceFrequency(&freq);					
    QueryPerformanceCounter(&s1);
    long s = s1.QuadPart/freq.QuadPart;
    long rm = s1.QuadPart % freq.QuadPart;
    long us = long(rm / (freq.QuadPart/1000000.0));
    
    tp->tv_sec = s;
    tp->tv_usec = us ;// us;
    return 0; 
}
#endif
