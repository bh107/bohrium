#ifndef __KP_CORE_TIMEVAULT_HPP
#define __KP_CORE_TIMEVAULT_HPP 1
#include <sys/time.h>
#include <stdint.h>
#include <string>
#include <map>

namespace kp{
namespace core{

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
        void store(std::string, time_t elapsed);

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
         *  Returns a textual representation of the elapsed time,
         *  control with or without detailed by calling TIMER_DETAILED macro
         *  or set_detailed(...);
         *
         *  TODO: Add average and deviation.
         */
        std::string text(void);

        /**
         *  Controls whether or not text(void) should return detailed dump.
         */
        void set_detailed(bool detailed);

        /**
         *  The de-constructor will dump-timings.
         */
        ~Timevault();

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

        std::map<std::string, std::vector<time_t> > _elapsed; // Storage of elapsed time

        time_t timer_start; // Storage for timer

        static const uint32_t width;
        bool _detailed;
};

}}

//
// Profiling macros for non-intrusive profiling.
//
#ifdef VE_CPU_PROFILING
#define TIMER_START               do{ kp::core::Timevault::instance().start(); } while(0);
#define TIMER_STOP(IDENT) do{ kp::core::Timevault::instance().store(IDENT, kp::core::Timevault::instance().stop()); } while(0);
#define TIMER_DUMP                do{ cout << kp::core::Timevault::instance().text() << endl; } while(0);
#define TIMER_DETAILED       do{ kp::core::Timevault::instance().set_detailed(true); } while(0);
#else
#define TIMER_START
#define TIMER_STOP(IDENT)
#define TIMER_DUMP
#define TIMER_DETAILED
#endif

#endif
