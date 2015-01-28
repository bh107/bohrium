#ifndef __BH_VE_CPU_THREAD_CONTROL
#define __BH_VE_CPU_THREAD_CONTROL
#include <string>

namespace bohrium{
namespace engine{
namespace cpu{

typedef enum {
    BIND_TO_NONE = 0,
    BIND_TO_CORE = 1,
    BIND_TO_PU = 2
} thread_binding;

class ThreadControl {
public:
    ThreadControl(thread_binding binding, size_t mthreads);

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
    size_t get_mthreads(void);

    ~ThreadControl(void);

    std::string text(void);

private:
    ThreadControl();

    thread_binding binding_;
    size_t mthreads_;
    static const char TAG[];
};

}}}
#endif
