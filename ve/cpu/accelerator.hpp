#ifndef __BH_VE_CPU_ACCELERATOR
#define __BH_VE_CPU_ACCELERATOR
#include <string>
#include <set>
#include "tac.h"

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
     *  Return amount of bytes allocated on accelerator.
     */
    size_t bytes_allocated(void);

    /**
     *  Check that operand-buffer is allocated on accelerator.
     */
    bool allocated(operand_t& operand);

    /**
     *  Check that operand-buffer has been pushed to the accelerator.
     */
    bool pushed(operand_t& operand);

    /**
     *  Allocate operand-buffer on accelerator.
     */
    void alloc(operand_t& operand);

    /**
     *  Free operand-buffer on accelerator.
     */
    void free(operand_t& operand);

    /**
     *  Push data from host to accelerator.
     */
    void push(operand_t& operand);

    /**
     *  Allocate operand-buffer on accelerator and
     *  push data from host to accelerator.
     */
    void push_alloc(operand_t& operand);

    /**
     *  Pull data from accelerator to host.
     */
    void pull(operand_t& operand);

    /**
     *  Pull data from accelerator to host and
     *  free operand-buffer on accelerator.
     */
    void pull_free(operand_t& operand);

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
    void _alloc(operand_t& operand);

    template <typename T>
    void _free(operand_t& operand);

    template <typename T>
    void _push(operand_t& operand);

    template <typename T>
    void _pull(operand_t& operand);

    int id_;
    size_t bytes_allocated_;
    std::set<const bh_base*> buffers_allocated_;

    std::set<const bh_base*> buffers_pushed_;

    static const char TAG[];
};

}}}

#endif
