#ifndef __BH_CORE_TIMEVAULT
#define __BH_CORE_TIMEVAULT
#include <sys/time.h>
#include <string>
#include <map>

namespace bohrium{
namespace core {

/**
 *  Sample and store wall-clock time.
 *  Implemented as a singleton.
 */
class Timevault {
    public:
        /**
         *  Returns the singleton instance of the Timevault.
         */
        static Timevault& instance();

        /**
         *  Returns current wall-clock time in microseconds.
         */
        static time_t sample_time(void);

        /**
         *  Start timer.
         */
        void start(void);

        /**
         *  Stop timer and return elapsed time.
         */
        time_t stop(void);

        /**
         *  Store elapsed time and associate it with the 'default' identifier.
         */
        void store(time_t elapsed);

        /**
         *  Store elapsed time and associate it with the given identifier.
         */
        void store(std::string identifier, time_t elapsed);

        /**
         *  Clears all stored time within the timevault.
         */
        void clear(void);

        /**
         *  Clears all time associated with identifier within in the timevault.
         */
        void clear(std::string identifier);

        /**
         *  Returns a string format for the given elapsed time.
         */
        template <typename T>
        static std::string format(T microseconds);

        /**
         *  Returns a string formatted with arguments as columns in a row.
         */
        static std::string format_row(std::string identifier, time_t elapsed, int samples);

        /**
         *  Return a string representing a row seperator.
         */
        static std::string format_line(char fill, char sep);

        /**
         *  Returns a textual representation of the elapsed time
         *  stored within the timevault, without details.
         *
         *  TODO: Add average and deviation.
         */
        std::string text(void);

        /**
         *  Returns a textual representation of the elapsed time
         *  stored within the timevault, with or without details.
         */
        std::string text(bool detailed);

        /**
         *  Writes the textual representation of the elapsed time
         *  stored within the Timevault to file.
         */
        void to_file(std::string absolute_path);

    private:
        /**
         * Non-Implemented: instantiation is controlled by instance().
         */
        Timevault();
        
        /**
         * Non-Implemented: instantiation is controlled by instance().
         */
        Timevault(Timevault const& copy);

        /**
         * Non-Implemented: instantiation is controlled by instance().
         */
        Timevault& operator=(Timevault const& copy);

        std::multimap<std::string, time_t> _elapsed; // Storage of elapsed time.

        time_t timer_start; // Storage for timer
};

}};

//
// Profiling macros for non-intrusive profiling.
//
#ifdef PROFILING
#define TIMER_START               do{ Timevault::instance().start(); } while(0);
#define TIMER_STOP(identifier)    do{ Timevault::instance().store(identifier, Timevault::instance().stop()); } while(0);
#define TIMER_DUMP                do{ cout << Timevault::instance().text() << endl; } while(0);
#define TIMER_DUMP_DETAILED       do{ cout << Timevault::instance().text(true) << endl; } while(0);
#else
#define TIMER_START
#define TIMER_STOP(identifier)
#define TIMER_DUMP
#define TIMER_DUMP_DETAILED
#endif

#endif
