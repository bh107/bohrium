#ifndef __BH_FILTER_COMPOSITE_EXPANDER
#define __BH_FILTER_COMPOSITE_EXPANDER
#include "bh.h"

namespace bohrium {
namespace filter {
namespace composite {

class Expander
{
public:
    Expander();

    /**
     *  Modifies the given bhir, expanding composites per configuration.
     */
    void expand(bh_ir& bhir);

    /**
     *  Expand sign at the given idx in the bhir instruction list.
     *
     *  Returns the number of additional instructions used.
     */
    int expand_sign(bh_ir& bhir, int idx);

    /**
     *  Expand matmul at the given idx in the bhir instruction list.
     *
     *  Returns the number of additional instructions used.
     */
    int expand_matmul(bh_ir& bhir, int idx);

private:

    static const char TAG[];
};

}}}
#endif
