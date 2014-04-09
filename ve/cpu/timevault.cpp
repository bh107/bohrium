#include "timevault.hpp"
#include <sstream>
#include <iostream>

using namespace std;
namespace bohrium{
namespace core {

Timevault::Timevault() {}

Timevault& Timevault::instance()
{
    static Timevault _instance;
    return _instance;
}

time_t Timevault::sample_time(void)
{
#ifndef _WIN32
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    return tv.tv_usec + tv.tv_sec * 1000000;
#else
    LARGE_INTEGER freq;
    LARGE_INTEGER s1;
    QueryPerformanceFrequency(&freq);                   
    QueryPerformanceCounter(&s1);
    long s  = s1.QuadPart / freq.QuadPart;
    long rm = s1.QuadPart % freq.QuadPart;
    long us = long(rm / (freq.QuadPart/1000000.0));
    return us + s * 1000000;
#endif
}

void Timevault::start(void)
{
    timer_start = sample_time();
}

time_t Timevault::stop(void)
{
    return sample_time() - timer_start;
}

void Timevault::store(time_t elapsed)
{
    store("default", elapsed);
}

void Timevault::store(string identifier, time_t elapsed)
{
    _elapsed.insert(std::pair<string, time_t>(identifier, elapsed));
}

void Timevault::clear(void)
{
    _elapsed.clear();
}

void Timevault::clear(string identifier)
{
    _elapsed.erase(identifier);
}

string Timevault::format(time_t elapsed)
{
    stringstream ss;
    ss << (elapsed/(1000.0*1000.0)) << " sec";
    return ss.str();
}

string Timevault::text(void)
{
    return text(false);
}

string Timevault::text(bool detailed)
{
    stringstream details, summary;

    time_t elapsed_total = 0;
    size_t samples_total = 0;

    //
    // Iterate over identifiers
    for(multimap<string, time_t>::iterator it=_elapsed.begin();
        it!=_elapsed.end();
        it=_elapsed.upper_bound(it->first)) {
        
        pair<multimap<string, time_t>::iterator, multimap<string, time_t>::iterator> ret;
        ret = _elapsed.equal_range((*it).first);

        //
        // Accumulate elapsed time for each identifier and count amount of samples
        time_t acc = 0;
        size_t samples = 0;
        for(multimap<string, time_t>::iterator inner=ret.first;
            inner!=ret.second;
            ++inner) {
            samples++;
            acc += (*inner).second;
            details << (*inner).first << "=" << (*inner).second << endl;
        }
        elapsed_total += acc;
        samples_total += samples;

        //
        // Create textual representation
        details << (*it).first << " = " << format(acc) <<  endl;
        summary << (*it).first << " = " << format(acc) << " over " << samples << " samples." << endl;
    }
    summary << "Total elapsed wall-clock: " << format(elapsed_total) << " over " << samples_total << " samples." << endl;

    if (detailed) {
        return details.str() + summary.str();
    } else {
        return summary.str();
    }
}

}}
