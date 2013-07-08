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

#ifdef _WIN32
#include <time.h>
#else
#include <sys/time.h>
#include <unistd.h>
#include <sys/types.h>
#endif

#include <bh_timing.h>
#include <assert.h>
#include <bh_component.h>
#include <map>
#include <list>
#include <iostream>
#include <fstream>
#include <sstream>

#ifdef _WIN32
    #include <Windows.h>
#endif

typedef struct
{
    bh_uint64 start;
    bh_uint64 end;
}interval;

typedef struct
{
    //The name of the timing
    char name[BH_COMPONENT_NAME_SIZE];
    //Number of timing intervals saved
    bh_intp count;
    //The total sum of the timing intervals
    bh_intp sum;
    //The timing intervals saved
    std::list<interval> *intervals;
}timing;

static std::map<bh_intp,timing> id2timing;
static bh_intp timer_count = 0;


/* Initiate new timer object.
 *
 * @name Name of the timing.
 * @return The timer ID.
 */
bh_intp _bh_timing_new(const char *name)
{
    timing t;
    strncpy(t.name, name, BH_COMPONENT_NAME_SIZE);
    t.intervals = new std::list<interval>();
    t.count = 0;
    t.sum = 0;
    id2timing.insert(std::pair<bh_intp,timing>(++timer_count, t));
    return timer_count;
}


/* Save a timing.
 *
 * @id     The ID of the timing.
 * @start  The start time in micro sec.
 * @end    The end time in micro sec.
 */
void _bh_timing_save(bh_intp id, bh_uint64 start, bh_uint64 end)
{
    assert(id2timing.find(id) != id2timing.end());
    timing *t = &id2timing[id];
    ++t->count;
    t->sum += end - start;
    const interval i = {start, end};
    t->intervals->push_back(i);
}


/* Save the sum of a timing.
 *
 * @id     The ID of the timing.
 * @start  The start time in micro sec.
 * @end    The end time in micro sec.
 */
void _bh_timing_save_sum(bh_intp id, bh_uint64 start, bh_uint64 end)
{
    assert(id2timing.find(id) != id2timing.end());
    timing *t = &id2timing[id];
    ++t->count;
    t->sum += end - start;
}


/* Get time.
 *
 * @return The current time.
 */
bh_uint64 _bh_timing(void)
{
#ifndef _WIN32
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    return (bh_uint64) tv.tv_usec +
           (bh_uint64) tv.tv_sec * 1000000;
#else
    LARGE_INTEGER freq;
    LARGE_INTEGER s1;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&s1);
    long s = s1.QuadPart/freq.QuadPart;
    long rm = s1.QuadPart % freq.QuadPart;
    long us = long(rm / (freq.QuadPart/1000000.0));
    return (bh_uint64) us + (bh_uint64) s * 1000000;
#endif
}

/* Dumps all timings to a file in the working directory.
 *
 */
void _bh_timing_dump_all(void)
{
    std::ofstream file;
    std::stringstream s, f;
    bh_intp pid;
#ifdef _WIN32
    pid = (bh_intp) GetCurrentProcessId();
#else
    pid = (bh_intp) getpid();
#endif
    char hname[1024];
    gethostname(hname, 1024);
    char fname[1024];
    sprintf(fname, "bh_stat.%s.%lld",hname,(long long)pid);

    s << "Timings from the execution (count):\n";
    for(std::map<bh_intp, timing>::iterator it=id2timing.begin();
        it!=id2timing.end(); ++it)
    {
        //Write to file
        f << it->second.name << ":\n";
        for(std::list<interval>::iterator it2=it->second.intervals->begin();
            it2!=it->second.intervals->end(); ++it2)
        {
            f << "\t" << it2->start << " > " << it2->end << "\n";
        }
        f << "\n";

        //Write resume to screen
        s << "\t" << it->second.name << ": \t" << it->second.sum << "us (" << it->second.count << ")\n";
        delete it->second.intervals;
    }
    s << "Writing timing details to file: " << fname << "\n";
    std::cout << s.str();

    file.open(fname);
    file << f.str();
    file.close();
}




