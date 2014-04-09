#include "timevault.hpp"
#include <sstream>
#include <iostream>
#include <iomanip>

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

template <typename T>
string Timevault::format(T microseconds)
{
    stringstream ss;

    ss.precision(4);
    ss << fixed << (microseconds/(1000.0*1000.0));

    return ss.str();
}

string Timevault::format_row(string identifier, time_t elapsed, int samples)
{
    stringstream ss;
    string sep = " | ";

    ss << setw(42)  << identifier << sep;
    ss << setw(7)   << samples << sep;
    ss << setw(10)  << format(elapsed) << sep;
    ss << setw(10)  << format(elapsed/(float)samples);

    return ss.str();
}

string Timevault::format_line(char fill, char sep)
{
    stringstream line;

    line << sep << setw(43) << setfill(fill) << sep;
    line << setw(10) << setfill(fill) << sep;
    line << setw(13) << setfill(fill) << sep;
    line << setw(13) << setfill(fill) << sep;
    line << endl;

    return line.str();
}

string Timevault::text(void)
{
    return text(false);
}

string Timevault::text(bool detailed)
{
    stringstream header, details, summary;

    time_t elapsed_total = 0;
    size_t samples_total = 0;

    header << endl;
    header << "  Identifier" << setw(32) << "|";
    header << " Samples " << "|";
    header << "  Elapsed   " << "|";
    header << "  Average " << endl;

    header << format_line('=', '+');

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
            details << format_row((*inner).first, (*inner).second, 1) << endl;
        }
        elapsed_total += acc;
        samples_total += samples;

        //
        // Create textual representation

        details << format_line('-', '+');
        details << format_row("Subtotal", acc, samples) << endl << endl;
        summary << format_row((*it).first, acc, samples) << endl;
    }
    summary << format_line('=', '+');
    summary << format_row("Total", elapsed_total, samples_total) << endl;

    if (detailed) {
        return header.str() + details.str() + header.str() + summary.str();
    } else {
        return header.str() + summary.str();
    }
}

}}
