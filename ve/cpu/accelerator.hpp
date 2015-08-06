#ifndef __BH_VE_CPU_ACCELERATOR
#define __BH_VE_CPU_ACCELERATOR
#include <string>
#include <set>
#include "kp_tac.h"

namespace bohrium{
namespace engine{
namespace cpu{

class Accelerator {
public:
    /**
     *  Construct instance wrapping an accelerator.
     *
     *  @param id Device id of the accelerator.
     *  @param offload Whether or not offloading to the device is enabled.
     */
    Accelerator(int id);

    /**
     *  Return device id associated with this accelerator instance.
     */
    int id(void);

    /**
     *  Returns a textual representation of the device.
     */
    std::string text(void);

    /**
     *  Determine whether the accelerator can be used.
     */
    bool offloadable(void);

    /**
     *  Return amount of bytes allocated on accelerator.
     */
    size_t bytes_allocated(void);

    /**
     *  Check that kp_operand-buffer is allocated on accelerator.
     */
    bool allocated(kp_operand & operand);

    /**
     *  Check that kp_operand-buffer has been pushed to the accelerator.
     */
    bool pushed(kp_operand & operand);

    /**
     *  Allocate kp_operand-buffer on accelerator.
     */
    void alloc(kp_operand & operand);

    /**
     *  Free kp_operand-buffer on accelerator.
     */
    void free(kp_operand & operand);

    /**
     *  Push data from host to accelerator.
     */
    void push(kp_operand & operand);

    /**
     *  Allocate kp_operand-buffer on accelerator and
     *  push data from host to accelerator.
     */
    void push_alloc(kp_operand & operand);

    /**
     *  Pull data from accelerator to host.
     */
    void pull(kp_operand & operand);

    /**
     *  Pull data from accelerator to host and
     *  free kp_operand-buffer on accelerator.
     */
    void pull_free(kp_operand & operand);

    /**
     *  Get max threads on accelerator.
     */
    int get_max_threads(void);

private:
    /**
     *  Construct accelerator with device id 0 and offload enabled.
     *
     *  NOTE: For now we don't want that...
     */
    Accelerator(void);

    template <typename T>
    void _alloc(kp_operand & operand);

    template <typename T>
    void _free(kp_operand & operand);

    template <typename T>
    void _push(kp_operand & operand);

    template <typename T>
    void _pull(kp_operand & operand);

    int id_;
    size_t bytes_allocated_;
    std::set<const bh_base*> buffers_allocated_;

    std::set<const bh_base*> buffers_pushed_;

    static const char TAG[];
};

}}}

#endif
