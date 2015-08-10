#ifndef __KP_ENGINE_THREAD_CONTROL_HPP
#define __KP_ENGINE_THREAD_CONTROL_HPP 1
#include <string>

namespace kp{
namespace engine{

typedef enum {
    BIND_TO_NONE = 0,
    BIND_TO_CORE = 1,
    BIND_TO_PU = 2
} thread_binding;

std::string cpu_text(void);
std::string coproc_text(void);

class ThreadControl {
public:
    ThreadControl(thread_binding binding, size_t thread_limit);

    /**
     *  Bind OpenMP threads to CORESs or PUs (HWLOC terminology).
     *
     *  NOTE: Caps the OpenMP work-group size to the number
     *        of objects(PUs or CORES).
     *
     *  @param binding Do nothing with BIND_TO_NONE, otherwise,
     *                  bind to PU or CORE.
     *  @return 0 on success or one of potentially multiple errors.
     */
    size_t bind_threads(thread_binding binding);
    size_t bind_threads();

    thread_binding get_binding(void);
    size_t get_thread_limit(void);

    ~ThreadControl(void);

    std::string text(void);

private:
    ThreadControl();

    thread_binding binding_;
    size_t thread_limit_;
    static const char TAG[];
};

}}

#endif
