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

#ifndef __BH_TIMING_HPP
#define __BH_TIMING_HPP

//When BH_TIMING_SUM is defined we only record timing sums.
#ifdef BH_TIMING_SUM
    #define BH_TIMING
#endif

#ifdef BH_TIMING

#include "bh_type.h"
#ifdef _WIN32
#include <Windows.h>
#include <time.h>
#else
#include <sys/time.h>
#endif
#include <iostream>
#include <mutex>
#include <vector>
#include <fstream>



namespace bh
{
    struct timing2 {bh_uint64 start; bh_uint64 end;};
    struct timing4 {bh_uint64 queued; bh_uint64 submit; bh_uint64 start; bh_uint64 end;};
}

std::ostream& operator<< (std::ostream& os, bh::timing2 const& t);
std::ostream& operator<< (std::ostream& os, bh::timing4 const& t);

namespace bh
{
    template <typename T=timing2, bh_uint64 GRANULARITY=1000000>
    class Timer
    {
    private:
        std::string name;
        bh_uint64 total;
        std::mutex mtx;
#ifndef BH_TIMING_SUM
        std::vector<T> values;
#endif
        Timer() {}
#ifndef BH_TIMING_SUM
        void write2file() const
        {
            std::ofstream file;
            file.open(name);
            for (const T& val: values)
            {
                file << val << std::endl;
            }
            file.close();
        }
#endif
        void print() const
        {
                std::cout << std::fixed;
                std::cout << "[Timing] " << name << ": " << (double)total / (double)GRANULARITY << std::endl;
        }
    public:
        Timer(std::string name_): name(name_), total(0) { }
        static bh_uint64 stamp()
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
        void add(T v)
        {
            mtx.lock();
            total += v.end - v.start;
#ifndef BH_TIMING_SUM
            values.push_back(v);
#endif
            mtx.unlock();
        }
        ~Timer()
        {
            print();
#ifndef BH_TIMING_SUM
            write2file();
#endif
        }
    };
}

#endif

//Only when BH_TIMING is defined will the bh_timing* functions do anything.
#ifdef BH_TIMING
    #define bh_timer_new(name) ((bh_intp)(new bh::Timer<>(name)))
    #define bh_timer_stamp() (bh::Timer<>::stamp())
    #define bh_timer_add(id,v1,v2) (((bh::Timer<>*)id)->add({v1,v2}))
    #define bh_timer_finalize(id) (delete (bh::Timer<>*)id)
#else
    #define bh_timer_new(name) ((bh_intp)0)
    #define bh_timer_stamp() ((bh_uint64)0)
    #define bh_timer_add(id,v1,v2) do{(void)(id);(void)(v1);(void)(v2);} while (0)
    #define bh_timer_finalize(id) do{(void)(id);} while (0)
#endif

#endif /* __BH_TIMING_HPP */
