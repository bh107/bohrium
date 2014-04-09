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
         *  Returns a string format for the given elapsed time.
         */
        static std::string format(time_t elapsed);

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
         *  Returns a textual representation of the elapsed time
         *  stored within the timevault, without details.
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
};

}};

// TODO: Add some convenient macros for profiling function-calls etc.

#endif
