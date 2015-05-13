#ifndef __BH_FILTER_COMPOSITE_EXPANDER
#define __BH_FILTER_COMPOSITE_EXPANDER
#include "bh.h"

namespace bohrium {
namespace filter {
namespace composite {

class Expander
{
public:
    /**
     *  Construct the expander.
     */
    Expander(void);

    /**
     *  Tear down the expander.
     *  This includes freeing allocated bh_base pointers.
     */
    ~Expander(void);

    /**
     *  Modifies the given bhir, expanding composites per configuration.
     */
    void expand(bh_ir& bhir);

    /**
     *  Collect garbage, that is de-allocate an amount of bh_base.
     *
     *  Make sure that BH_FREE and BH_DISCARD has been sent down the stack.
     *
     *  It seems likely that re-use would be an idea so default
     *  strategy should probably be to keep some amount for re-use
     *  and de-allocate above some threshold.
     *
     *  Upon deconstruction everything should of course be de-allocated.
     */
    int gc(void);

    /**
     *  Expand BH_SIGN at the given idx into the sequence:
     *
     *      BH_SIGN OUT, IN
     *
     *      LESS T1, IN, 0
     *      GREATER T2, IN, 0
     *      SUBTRACT OUT, T1, T2
     *      FREE T1
     *      DISCARD T1
     *      FREE T2
     *      DISCARD T2
     *
     *  Returns the number of additional instructions used (6).
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
    std::vector<bh_base*> bases_;
};

}}}
#endif
